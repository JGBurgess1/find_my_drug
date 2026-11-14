"""
Build an FPSim2 fingerprint database from a ZINC20 directory tree.

Layout expected (one layer of subdirectories, several multi-SMILES files each):

    ZINC20/
        AAAA/
            AAAAML.smi
            AAAARN.smi
        AABA/
            AABAML.smi
            ...

Each .smi line is '<SMILES> <zinc_id>', e.g.

    O=C(NCc1ccc(N2CCOCC2)c(F)c1)N1CCC(c2cccc3ccccc23)CC1 58571844

Fingerprints: Morgan, radius 3, 2048 bits (FPSim2's own Morgan defaults).


Parallelism
───────────
Three levels, which compose:

  1. MPI across nodes/ranks  — each rank builds its own SHARD of the database
  2. multiprocessing per rank — fingerprinting inside one shard, via the
                                fpsim2-create-db CLI's --processes
  3. nothing else; RDKit itself is single-threaded here

FPSim2 writes one .h5 through PyTables and offers no way to merge two
databases, so ranks cannot cooperatively build a single file. Instead each
rank writes an independent shard and a manifest ties them together;
fpsim2_similarity_search.py fans a query out over the shards and merges the
hits. That is not a workaround so much as an advantage at ZINC20 scale: a
single .h5 of ~260 bytes/compound would be far too large to hold in one
node's RAM, while an individual shard comfortably fits, so searches run in
memory rather than streaming from disk.

MPI mode is detected automatically (mpi4py present and COMM_WORLD size > 1).
Use one rank per node with --processes set to that node's core count; with
many ranks per node, leave --processes at its MPI default of 1 to avoid
oversubscription.


Artefacts
─────────
  <out>.manifest.json          — shard inventory; pass THIS to the search script
  <out>.h5 / .sqlite           — single-rank build
  <out>.shardNNN.h5 / .sqlite  — one pair per rank under MPI

The .sqlite sidecars map zinc_id -> SMILES. They are needed because FPSim2
stores only integer molecule ids and fingerprints, never structures: a search
returns ids, so the SMILES has to be looked up separately. A sidecar is a
superset of its .h5 — it keeps every parseable line, while FPSim2 drops
entries RDKit cannot turn into a molecule.

Build stages, per rank:

    discover -> sidecar sqlite (dedupes by id) -> one .smi -> FPSim2 h5

The intermediate .smi is written because FPSim2's own reader
(smi_mol_supplier) takes a single file, splits each line with rsplit(None, 1),
requires an integer id and tolerates no header. Normalising once up front
means bad lines are reported here rather than aborting a multi-hour
fingerprint run. Routing it through sqlite deduplicates repeated zinc_ids on
disk at no memory cost.

Note that deduplication is per shard: the same compound appearing in two
shards is fingerprinted twice. The search script dedupes hits by zinc_id when
merging, so this costs a little disk, not correctness.

Usage:
    # single node, all cores
    python fpsim2_build_db.py -i /path/to/ZINC20 -o zinc20_morgan_r3_2048 \
        --processes 24

    # 10 nodes x 24 cores, one rank per node
    mpiexec -n 10 python fpsim2_build_db.py \
        -i /path/to/ZINC20 -o zinc20_morgan_r3_2048 --processes 24

    # 240 ranks, one core each
    mpiexec -n 240 python fpsim2_build_db.py \
        -i /path/to/ZINC20 -o zinc20_morgan_r3_2048

    # smoke test on two subdirectories
    python fpsim2_build_db.py -i /path/to/ZINC20 -o test --limit-subdirs 2
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

FP_TYPE = "Morgan"
# FPSim2's Morgan defaults are already radius 3 / fpSize 2048; set explicitly
# so the database records the intended parameters rather than inheriting them.
FP_PARAMS = {"radius": 3, "fpSize": 2048}

# FPSim2 returns results as a structured array with dtype
# [("mol_id", "<u4"), ("coeff", "<f4")] — mol_id is uint32, so an id above
# this is returned truncated modulo 2**32. Verified: a compound stored with
# id 5000000000 searches back as 705032704, with no error.
UINT32_MAX = 4_294_967_295

SIDECAR_SCHEMA = """
CREATE TABLE IF NOT EXISTS compounds (
    zinc_id INTEGER PRIMARY KEY,
    smiles  TEXT    NOT NULL,
    source  TEXT
);
"""

log = logging.getLogger(__name__)


# ── MPI ───────────────────────────────────────────────────────────────────────

def init_mpi(disable: bool = False):
    """
    Return (comm, rank, size). Falls back to (None, 0, 1) when mpi4py is
    absent, when launched outside mpiexec, or when --no-mpi is given.
    """
    if disable:
        return None, 0, 1
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
    except Exception as exc:
        # Deliberately broad: importing mpi4py raises RuntimeError ("cannot
        # load MPI library"), not ImportError, when the package is installed
        # but no MPI runtime is present. Either way, run serially.
        logging.getLogger(__name__).debug("MPI unavailable (%s); serial", exc)
        return None, 0, 1
    if size == 1:
        return None, 0, 1
    return comm, comm.Get_rank(), size


def configure_logging(rank: int, size: int) -> None:
    """Tag every line with the rank so interleaved shard logs stay readable."""
    prefix = f"[rank {rank:03d}/{size:03d}] " if size > 1 else ""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s {prefix}[%(levelname)s] %(message)s",
    )


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_files(input_dir: str, file_glob: str = "*.smi",
                   limit_subdirs: int = 0) -> List[str]:
    """
    Scan one level of child subdirectories and return sorted file paths.
    Files sitting directly in input_dir are ignored.

    Mirrors discover_files() in predict_batch.py so both pipelines agree on
    what the ZINC20 layout looks like.
    """
    subdirs = sorted(
        d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)
    )
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found inside '{input_dir}'")

    if limit_subdirs:
        subdirs = subdirs[:limit_subdirs]
        log.info("Limited to the first %d subdirectory/ies", len(subdirs))

    all_files: List[str] = []
    for subdir in subdirs:
        all_files.extend(sorted(glob.glob(os.path.join(subdir, file_glob))))

    if not all_files:
        raise FileNotFoundError(
            f"No '{file_glob}' files found in any subdirectory of '{input_dir}'"
        )

    total_bytes = sum(os.path.getsize(f) for f in all_files)
    log.info(
        "Discovered %d file(s) across %d subdirectory/ies — %.2f GB",
        len(all_files), len(subdirs), total_bytes / 1e9,
    )
    return all_files


def assign_shard(paths: List[str], rank: int, size: int) -> List[str]:
    """
    Pick this rank's share of the input files.

    Files are sorted largest-first and dealt round-robin, which balances the
    load far better than contiguous slicing: ZINC20 tranche files vary by
    orders of magnitude in size, so a contiguous block can leave one rank
    fingerprinting for hours after the rest have finished.
    """
    if size == 1:
        return paths
    by_size = sorted(paths, key=lambda p: os.path.getsize(p), reverse=True)
    return by_size[rank::size]


# ── Normalisation ─────────────────────────────────────────────────────────────

def iter_records(paths: List[str]) -> Iterator[Tuple[str, int, str]]:
    """
    Stream (smiles, zinc_id, source_basename) over every input file.

    Lines are parsed the way FPSim2 does it — rsplit(None, 1), so a SMILES
    containing spaces still works — but malformed lines are skipped with a
    counter instead of raising. Nothing is held in memory beyond one line,
    which matters at ZINC20 scale.
    """
    stats = {"lines": 0, "bad_fields": 0, "bad_id": 0}
    for path in paths:
        source = os.path.basename(path)
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                stats["lines"] += 1
                fields = line.rsplit(None, 1)
                if len(fields) != 2:
                    stats["bad_fields"] += 1
                    continue
                smiles, raw_id = fields
                try:
                    zinc_id = int(raw_id)
                except ValueError:
                    # Covers header rows ('smiles zinc_id') and ZINC-prefixed
                    # ids ('ZINC000058571844'), which we salvage by stripping
                    # non-digits.
                    digits = "".join(c for c in raw_id if c.isdigit())
                    if not digits:
                        stats["bad_id"] += 1
                        continue
                    zinc_id = int(digits)
                yield smiles, zinc_id, source

    log.info(
        "Read %s line(s): %s skipped for field count, %s for unparseable id",
        f"{stats['lines']:,}", f"{stats['bad_fields']:,}",
        f"{stats['bad_id']:,}",
    )


def count_over_uint32(sidecar_path: Path) -> int:
    """Count sidecar ids that cannot survive FPSim2's uint32 result dtype."""
    conn = sqlite3.connect(str(sidecar_path))
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM compounds WHERE zinc_id > ?", (UINT32_MAX,)
        ).fetchone()[0]
    finally:
        conn.close()


def normalise(paths: List[str], smi_path: Path,
              sidecar_path: Path) -> Tuple[int, int]:
    """
    Load every record into the sidecar sqlite index, then dump it back out as
    a single FPSim2-ready '<smiles> <int_id>' file.

    Returns (compounds_written, largest_zinc_id).
    """
    if sidecar_path.exists():
        sidecar_path.unlink()
    conn = sqlite3.connect(str(sidecar_path))
    # Build-time speed settings; this file is rebuildable, so durability
    # during the build is not worth the write cost.
    conn.executescript("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;")
    conn.executescript(SIDECAR_SCHEMA)

    read = 0
    max_id = 0
    batch: List[Tuple[int, str, str]] = []

    # OR IGNORE keeps the first occurrence of a repeated id and makes the
    # duplicate count fall out of the row count for free.
    insert = ("INSERT OR IGNORE INTO compounds (zinc_id, smiles, source) "
              "VALUES (?, ?, ?)")

    t0 = time.time()
    for smiles, zinc_id, source in iter_records(paths):
        batch.append((zinc_id, smiles, source))
        read += 1
        max_id = max(max_id, zinc_id)

        if len(batch) >= 100_000:
            conn.executemany(insert, batch)
            batch.clear()
            if read % 1_000_000 == 0:
                rate = read / max(time.time() - t0, 1e-9)
                log.info("  indexed %s compounds (%s/s)",
                         f"{read:>13,}", f"{rate:,.0f}")

    if batch:
        conn.executemany(insert, batch)
    conn.commit()

    written = conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
    duplicates = read - written
    if duplicates:
        log.info("Dropped %s duplicate zinc_id(s)", f"{duplicates:,}")

    # Dump in primary-key order; FPSim2 re-sorts by popcount anyway.
    t1 = time.time()
    with open(smi_path, "w") as out:
        for smiles, zinc_id in conn.execute(
                "SELECT smiles, zinc_id FROM compounds ORDER BY zinc_id"):
            out.write(f"{smiles} {zinc_id}\n")
    log.info("Dumped FPSim2 input in %.1f s", time.time() - t1)

    conn.close()

    log.info("Normalised %s compounds -> %s", f"{written:,}", smi_path)
    log.info("Sidecar index -> %s (%.1f MB)",
             sidecar_path, sidecar_path.stat().st_size / 1e6)
    return written, max_id


# ── Fingerprinting ────────────────────────────────────────────────────────────

def build_with_cli(smi_path: Path, h5_path: Path, processes: int) -> bool:
    """
    Build via the fpsim2-create-db console script, which fingerprints in
    parallel. The Python API is single-threaded, so this is the route worth
    taking for anything ZINC20-sized. Returns False if the CLI is absent.
    """
    exe = shutil.which("fpsim2-create-db")
    if exe is None:
        log.info("fpsim2-create-db not on PATH; falling back to the Python API")
        return False

    cmd = [
        exe, str(smi_path), str(h5_path),
        "--fp_type", FP_TYPE,
        "--fp_params", json.dumps(FP_PARAMS),
        "--processes", str(processes),
    ]
    log.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(
            f"fpsim2-create-db exited {proc.returncode}; see output above"
        )
    return True


def build_with_api(smi_path: Path, h5_path: Path) -> None:
    """Build via FPSim2's Python API (single-threaded fingerprinting)."""
    try:
        from FPSim2.io import create_db_file
    except ImportError as exc:
        raise SystemExit(
            "ERROR: FPSim2 is not importable.\n"
            "  conda install -c conda-forge fpsim2\n"
            f"  (import failed with: {exc})"
        )

    create_db_file(
        mols_source=str(smi_path),
        filename=str(h5_path),
        mol_format=None,          # None for .smi/.sdf file paths
        fp_type=FP_TYPE,
        fp_params=FP_PARAMS,
    )


# ── Orchestration ─────────────────────────────────────────────────────────────

def shard_paths(out: str, idx: int, n_shards: int) -> Tuple[Path, Path, Path]:
    """Return (h5, sqlite, normalised_smi) for one shard."""
    stem = out if n_shards == 1 else f"{out}.shard{idx:03d}"
    return Path(f"{stem}.h5"), Path(f"{stem}.sqlite"), \
        Path(f"{stem}.normalised.smi")


def normalise_one_shard(all_paths: List[str], idx: int, n_shards: int,
                        args, comm) -> Optional[dict]:
    """
    Phase A for one shard: assign files, build the sidecar and the FPSim2
    input file. Returns a shard record, or None if no files fell to it.

    Normalising every shard before any fingerprinting starts is what lets the
    uint32 check below fail the job in minutes rather than after hours of
    fingerprint work.
    """
    h5_path, sidecar_path, smi_path = shard_paths(args.out, idx, n_shards)

    if h5_path.exists():
        abort(comm, f"ERROR: {h5_path} already exists. Move or delete it "
                    "first — FPSim2 will not append to an existing database.")

    my_paths = assign_shard(all_paths, idx, n_shards)
    if not my_paths:
        # More shards than input files. Harmless, but under MPI this rank must
        # still take part in the collectives that follow.
        log.warning("Shard %d: no input files assigned; skipping", idx)
        return None

    log.info("Shard %d/%d: %d file(s), %.2f GB", idx, n_shards, len(my_paths),
             sum(os.path.getsize(p) for p in my_paths) / 1e9)

    count, max_id = normalise(my_paths, smi_path, sidecar_path)
    return {
        "shard": idx,
        "h5": h5_path.name,
        "sqlite": sidecar_path.name,
        "compounds": count,
        "files": len(my_paths),
        # Not written to the manifest; dropped after the checks below.
        "_max_id": max_id,
        "_paths": (h5_path, sidecar_path, smi_path),
    }


def fingerprint_one_shard(record: dict, processes: int, args) -> None:
    """Phase B for one shard: turn its .smi into the FPSim2 .h5."""
    h5_path, _sidecar_path, smi_path = record["_paths"]
    idx = record["shard"]

    t0 = time.time()
    if args.api or not build_with_cli(smi_path, h5_path, processes):
        log.info("Fingerprinting with the Python API — single-threaded")
        build_with_api(smi_path, h5_path)
    log.info("Shard %d fingerprinted in %.1f min", idx, (time.time() - t0) / 60)

    if not args.keep_normalised:
        smi_path.unlink(missing_ok=True)


def abort(comm, message: str) -> None:
    """
    Fail the whole job, not just this rank. A rank raising SystemExit under
    mpiexec can leave the others spinning until walltime, so call Abort.
    """
    log.error(message)
    if comm is not None:
        comm.Abort(1)
    raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an FPSim2 Morgan r3/2048 database from a ZINC20 tree"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="ZINC20 root directory (one layer of subdirs)")
    parser.add_argument("--out", "-o", required=True,
                        help="Output basename; writes <out>.manifest.json plus "
                             "the shard .h5/.sqlite files")
    parser.add_argument("--file-glob", default="*.smi",
                        help="Glob for SMILES files inside each subdir "
                             "(default: *.smi)")
    parser.add_argument("--processes", "-p", type=int, default=0,
                        help="Fingerprinting processes per rank, via the "
                             "fpsim2-create-db CLI. Default: all cores when "
                             "serial, 1 under MPI (avoids oversubscription)")
    parser.add_argument("--limit-subdirs", type=int, default=0,
                        help="Only use the first N subdirectories (smoke test)")
    parser.add_argument("--allow-large-ids", action="store_true",
                        help="Build even if some zinc_id exceeds uint32; such "
                             "ids come back truncated from a search")
    parser.add_argument("--keep-normalised", action="store_true",
                        help="Keep the concatenated .smi files after the build")
    parser.add_argument("--api", action="store_true",
                        help="Force the single-threaded Python API even if the "
                             "CLI is available")
    parser.add_argument("--shards", type=int, default=1,
                        help="Serial mode only: split the build into N shards "
                             "and build them one after another. Useful for a "
                             "PBS job array, or to keep each shard small "
                             "enough to search in memory. Under MPI the shard "
                             "count is the rank count")
    parser.add_argument("--no-mpi", action="store_true",
                        help="Ignore MPI even when launched under mpiexec")
    args = parser.parse_args()

    comm, rank, size = init_mpi(disable=args.no_mpi)
    configure_logging(rank, size)

    processes = args.processes or (1 if size > 1 else (os.cpu_count() or 1))
    if rank == 0 and size > 1:
        log.info("MPI build across %d rank(s), %d fingerprinting process(es) "
                 "each", size, processes)

    manifest_path = Path(f"{args.out}.manifest.json")

    # Which shards exist, and which of them are mine.
    #   under MPI : one shard per rank, this rank builds its own
    #   serially  : --shards N shards, built one after another here
    if size > 1:
        n_shards, my_shards = size, [rank]
        if args.shards and rank == 0:
            log.warning("--shards %d ignored under MPI; using one shard per "
                        "rank (%d)", args.shards, size)
    else:
        n_shards = max(args.shards, 1)
        my_shards = list(range(n_shards))

    # ── Discovery on rank 0, then broadcast ──────────────────────────────────
    all_paths: Optional[List[str]] = None
    if rank == 0:
        try:
            all_paths = discover_files(args.input, args.file_glob,
                                       args.limit_subdirs)
        except FileNotFoundError as exc:
            abort(comm, f"ERROR: {exc}")
    if comm is not None:
        all_paths = comm.bcast(all_paths, root=0)

    # ── Phase A: normalise every shard ───────────────────────────────────────
    records = [
        r for r in (
            normalise_one_shard(all_paths, idx, n_shards, args, comm)
            for idx in my_shards
        ) if r is not None
    ]

    # ── Collective uint32 check, before any fingerprinting ───────────────────
    # Decided globally so every rank aborts together rather than some building
    # a database the search step cannot resolve.
    global_max = max((r["_max_id"] for r in records), default=0)
    n_over = sum(count_over_uint32(r["_paths"][1]) for r in records)
    if comm is not None:
        from mpi4py import MPI
        global_max = comm.allreduce(global_max, op=MPI.MAX)
        n_over = comm.allreduce(n_over, op=MPI.SUM)

    if global_max > UINT32_MAX and not args.allow_large_ids:
        abort(comm, (
            f"ERROR: {n_over:,} zinc_id(s) exceed uint32 ({UINT32_MAX:,}), "
            f"the largest being {global_max:,}.\n"
            "FPSim2 returns results as dtype [('mol_id','<u4'),('coeff','<f4')],"
            " so such ids come back truncated modulo 2**32 — verified: "
            "5000000000 searched back as 705032704. Hits would carry wrong ids "
            "and silently fail to resolve against the sidecar.\n"
            "Re-run with --allow-large-ids to build anyway (the search script "
            "reports hits whose id is missing from the sidecar), or split the "
            "input so every id fits in uint32."
        ))

    # ── Phase B: fingerprint ─────────────────────────────────────────────────
    for record in records:
        fingerprint_one_shard(record, processes, args)

    # ── Manifest on rank 0 ───────────────────────────────────────────────────
    for record in records:            # strip the internal bookkeeping keys
        record.pop("_max_id", None)
        record.pop("_paths", None)

    entries = [records] if comm is None else comm.gather(records, root=0)

    if rank == 0:
        shards = sorted((e for part in entries for e in part),
                        key=lambda e: e["shard"])
        total = sum(e["compounds"] for e in shards)
        if total == 0:
            abort(comm, "ERROR: no usable compounds found; nothing was built")

        manifest = {
            "fp_type": FP_TYPE,
            "fp_params": FP_PARAMS,
            "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": os.path.abspath(args.input),
            "total_compounds": total,
            # Paths are stored as bare names and resolved against the
            # manifest's own directory, so the set can be moved or copied
            # between filesystems without rewriting it.
            "shards": shards,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

        log.info("")
        log.info("Shards          : %d", len(shards))
        log.info("Total compounds : %s", f"{total:,}")
        log.info("Manifest        : %s", manifest_path)
        log.info(
            "\nNext step:\n"
            "  python fpsim2_similarity_search.py \\\n"
            "      --db %s --query '<SMILES>' --threshold 0.7 \\\n"
            "      --output hits.csv\n"
            "  (or: mpiexec -n %d python fpsim2_similarity_search.py ...)",
            manifest_path, max(len(shards), 1),
        )


if __name__ == "__main__":
    main()
