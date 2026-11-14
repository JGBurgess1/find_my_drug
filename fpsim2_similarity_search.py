"""
Tanimoto similarity search against an FPSim2 database built by
fpsim2_build_db.py.

Takes one query SMILES or a file of them, and returns every database compound
whose Tanimoto (Jaccard) coefficient against the query is >= a threshold.

Results carry the SMILES of each hit, looked up from the .sqlite sidecars the
build step wrote — FPSim2 itself stores only integer ids and fingerprints, so
a raw search result is ids and coefficients alone.


Sharded and MPI search
──────────────────────
A build may produce one database or many shards (see fpsim2_build_db.py).
Pass the manifest and this script handles either:

    --db zinc20_morgan_r3_2048.manifest.json

Under mpiexec, shards are dealt round-robin across ranks; each rank searches
its own shards, resolves SMILES from the matching sidecar, and sends hits to
rank 0, which merges and writes the CSV. Without MPI the same shards are
searched one after another in a single process.

Merging dedupes by zinc_id, keeping the best coefficient — the build dedupes
within a shard but not across them, so a compound present in two shards would
otherwise appear twice.

Pointing --db straight at a single .h5 also works, for a database built
before manifests or by hand.

Search modes:
  in-memory (default) — loads a shard's fingerprints into RAM; fastest.
                        Sharding is what makes this viable at ZINC20 scale:
                        one shard per rank fits where the whole database
                        would not.
  --on-disk           — streams instead, for a shard larger than RAM

Note on thresholds: FPSim2 uses the threshold to prune work via popcount
bounds, so it is dramatically faster at >= 0.7 than at low thresholds. A
search at 0.3 over a large database may take a very long time and return an
enormous result set.

Usage:
    # single query against a sharded build
    python fpsim2_similarity_search.py \
        --db zinc20_morgan_r3_2048.manifest.json \
        --query 'O=C(NCc1ccc(N2CCOCC2)c(F)c1)N1CCC(c2cccc3ccccc23)CC1' \
        --threshold 0.7 \
        --output hits.csv

    # many queries, one rank per shard
    mpiexec -n 10 python fpsim2_similarity_search.py \
        --db zinc20_morgan_r3_2048.manifest.json \
        --query-file actives.smi \
        --threshold 0.6 \
        --output hits.csv
"""

import argparse
import csv
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# FPSim2 >= 0.5 exposes 'tanimoto' and 'dice'. Tanimoto on binary
# fingerprints is the Jaccard index.
METRICS = ("tanimoto", "dice")

log = logging.getLogger(__name__)


# ── MPI ───────────────────────────────────────────────────────────────────────

def init_mpi(disable: bool = False):
    """Return (comm, rank, size); (None, 0, 1) when MPI is unavailable."""
    if disable:
        return None, 0, 1
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
    except Exception as exc:
        # Broad by design: mpi4py raises RuntimeError, not ImportError, when
        # installed without an MPI runtime behind it.
        logging.getLogger(__name__).debug("MPI unavailable (%s); serial", exc)
        return None, 0, 1
    if size == 1:
        return None, 0, 1
    return comm, comm.Get_rank(), size


def configure_logging(rank: int, size: int) -> None:
    prefix = f"[rank {rank:03d}/{size:03d}] " if size > 1 else ""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s {prefix}[%(levelname)s] %(message)s",
    )


# ── Shard discovery ───────────────────────────────────────────────────────────

def load_shards(db_arg: str) -> List[Tuple[Path, Optional[Path]]]:
    """
    Resolve --db into a list of (h5_path, sidecar_path_or_None).

    Accepts a manifest written by fpsim2_build_db.py, or a bare .h5 (whose
    sidecar is then guessed by swapping the suffix).
    """
    path = Path(db_arg)
    if not path.exists():
        raise SystemExit(f"ERROR: not found: {path}")

    if path.suffix == ".json":
        manifest = json.loads(path.read_text())
        base = path.parent
        shards = []
        for entry in manifest.get("shards", []):
            h5 = base / entry["h5"]
            sidecar = base / entry["sqlite"]
            if not h5.exists():
                raise SystemExit(
                    f"ERROR: shard {h5} listed in {path} is missing"
                )
            shards.append((h5, sidecar if sidecar.exists() else None))
        if not shards:
            raise SystemExit(f"ERROR: {path} lists no shards")
        log.info("Manifest %s: %d shard(s), %s compound(s), %s %s",
                 path.name, len(shards),
                 f"{manifest.get('total_compounds', 0):,}",
                 manifest.get("fp_type", "?"), manifest.get("fp_params", {}))
        return shards

    sidecar = path.with_suffix(".sqlite")
    return [(path, sidecar if sidecar.exists() else None)]


# ── Query loading ─────────────────────────────────────────────────────────────

def load_queries(query: Optional[List[str]], query_file: Optional[str]
                 ) -> List[Tuple[str, str]]:
    """
    Build a list of (query_name, smiles).

    --query values are named q1, q2, ... in order. A --query-file is read as
    '<SMILES> [name]' per line; the name defaults to its line position. Blank
    lines and '#' comments are skipped.
    """
    queries: List[Tuple[str, str]] = []

    for i, smi in enumerate(query or [], start=1):
        queries.append((f"q{i}", smi))

    if query_file:
        with open(query_file, "r") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.rsplit(None, 1)
                if len(fields) == 2:
                    smi, name = fields
                else:
                    smi, name = line, f"line{lineno}"
                queries.append((name, smi))

    return queries


def validate_queries(queries: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Drop queries RDKit cannot parse, before any searching starts — otherwise a
    typo in query 40 of 50 surfaces only after the earlier searches have run.
    """
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        log.warning("RDKit not importable; skipping query validation")
        return queries

    good = []
    for name, smi in queries:
        if Chem.MolFromSmiles(smi) is None:
            log.warning("Query %r is not valid SMILES, skipping: %s", name, smi)
        else:
            good.append((name, smi))
    return good


# ── Sidecar lookup ────────────────────────────────────────────────────────────

def lookup_smiles(sidecar: Optional[Path], mol_ids: List[int]
                  ) -> Dict[int, str]:
    """
    Map zinc_id -> SMILES for the hit ids, via a sidecar sqlite index.
    Returns an empty dict when there is no sidecar, so the caller can still
    emit ids and coefficients.
    """
    if sidecar is None or not sidecar.exists() or not mol_ids:
        return {}

    out: Dict[int, str] = {}
    # SQLite URI filenames need forward slashes, so as_posix() rather than
    # str() — 'file:C:\...' is not a valid URI. Fall back to a plain
    # connection if the read-only URI is rejected.
    try:
        conn = sqlite3.connect(f"file:{sidecar.as_posix()}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        conn = sqlite3.connect(str(sidecar))
    try:
        # Chunked IN (...) keeps us under SQLite's variable limit (999 by
        # default) and avoids building one enormous query.
        for start in range(0, len(mol_ids), 900):
            chunk = mol_ids[start:start + 900]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT zinc_id, smiles FROM compounds "
                f"WHERE zinc_id IN ({placeholders})",
                chunk,
            )
            out.update(rows)
    finally:
        conn.close()
    return out


# ── Search ────────────────────────────────────────────────────────────────────

def search_shard(h5: Path, sidecar: Optional[Path],
                 queries: List[Tuple[str, str]], args
                 ) -> List[Tuple[str, int, str, float]]:
    """
    Run every query against one shard.

    Returns (query_name, zinc_id, smiles, coeff) rows. The engine is built
    and dropped per shard so a rank holding several shards never has more
    than one in memory at a time.
    """
    from FPSim2 import FPSim2Engine

    t0 = time.time()
    fpe = FPSim2Engine(str(h5), in_memory_fps=not args.on_disk)
    log.info("  %s loaded in %.1f s (%.2f GB)%s", h5.name, time.time() - t0,
             h5.stat().st_size / 1e9, " on-disk" if args.on_disk else "")

    rows: List[Tuple[str, int, str, float]] = []
    for name, smi in queries:
        t_q = time.time()
        if args.on_disk:
            results = fpe.on_disk_similarity(
                smi, threshold=args.threshold, metric=args.metric,
                n_workers=args.n_workers, chunk_size=args.chunk_size,
            )
        else:
            results = fpe.similarity(
                smi, threshold=args.threshold, metric=args.metric,
                n_workers=args.n_workers,
            )

        # Structured array, dtype [("mol_id","<u4"), ("coeff","<f4")],
        # already sorted by coefficient descending. Trim per shard as well as
        # globally, so a huge low-threshold result set is not carried around
        # whole just to be cut later.
        if args.top_n:
            results = results[:args.top_n]

        mol_ids = [int(i) for i in results["mol_id"]]
        smiles_map = lookup_smiles(sidecar, mol_ids) if not args.no_smiles \
            else {}

        for mol_id, coeff in zip(mol_ids, results["coeff"]):
            rows.append((name, mol_id, smiles_map.get(mol_id, ""),
                         float(coeff)))

        log.info("    %-16s %8s hit(s) in %6.1f s", name,
                 f"{len(mol_ids):,}", time.time() - t_q)

    del fpe
    return rows


def merge_hits(rows: List[Tuple[str, int, str, float]], top_n: int
               ) -> Dict[str, List[Tuple[int, str, float]]]:
    """
    Group rows by query and dedupe by zinc_id, keeping the best coefficient.

    Needed because the build dedupes within a shard but not across shards, so
    a compound in two shards arrives twice.
    """
    best: Dict[str, Dict[int, Tuple[str, float]]] = {}
    for name, mol_id, smiles, coeff in rows:
        per_query = best.setdefault(name, {})
        prev = per_query.get(mol_id)
        if prev is None or coeff > prev[1]:
            per_query[mol_id] = (smiles, coeff)

    merged: Dict[str, List[Tuple[int, str, float]]] = {}
    for name, hits in best.items():
        ordered = sorted(
            ((mol_id, smiles, coeff)
             for mol_id, (smiles, coeff) in hits.items()),
            key=lambda r: (-r[2], r[0]),
        )
        merged[name] = ordered[:top_n] if top_n else ordered
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tanimoto similarity search over an FPSim2 database"
    )
    parser.add_argument("--db", "-d", required=True,
                        help="<out>.manifest.json from fpsim2_build_db.py, or "
                             "a single .h5")
    parser.add_argument("--query", "-q", action="append", metavar="SMILES",
                        help="Query SMILES; repeat for several queries")
    parser.add_argument("--query-file", "-Q",
                        help="File of '<SMILES> [name]' lines")
    parser.add_argument("--threshold", "-t", type=float, required=True,
                        help="Minimum Tanimoto/Jaccard coefficient, e.g. 0.7")
    parser.add_argument("--metric", "-m", default="tanimoto", choices=METRICS,
                        help="Similarity metric (default: tanimoto)")
    parser.add_argument("--output", "-o", default="similarity_hits.csv",
                        help="Output CSV (default: similarity_hits.csv)")
    parser.add_argument("--on-disk", action="store_true",
                        help="Stream each shard from disk instead of loading "
                             "it into RAM")
    parser.add_argument("--n-workers", "-w", type=int, default=1,
                        help="Threads per query within a rank (default: 1)")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="On-disk read chunk size; 0 lets FPSim2 choose")
    parser.add_argument("--top-n", type=int, default=0,
                        help="Keep only the N best hits per query (0 = all)")
    parser.add_argument("--no-smiles", action="store_true",
                        help="Skip the sidecar lookup and emit ids only")
    parser.add_argument("--no-mpi", action="store_true",
                        help="Ignore MPI even when launched under mpiexec")
    args = parser.parse_args()

    comm, rank, size = init_mpi(disable=args.no_mpi)
    configure_logging(rank, size)

    if not 0.0 < args.threshold <= 1.0:
        raise SystemExit("ERROR: --threshold must be in (0, 1]")
    if args.threshold < 0.5 and rank == 0:
        log.warning(
            "Threshold %.2f is low. FPSim2 prunes candidates using popcount "
            "bounds derived from the threshold, so low values are much slower "
            "and can return very large result sets.", args.threshold,
        )

    queries = load_queries(args.query, args.query_file)
    if not queries:
        raise SystemExit("ERROR: no queries given; pass --query or --query-file")
    queries = validate_queries(queries)
    if not queries:
        raise SystemExit("ERROR: no valid queries remain after validation")

    shards = load_shards(args.db)

    try:
        import FPSim2  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ERROR: FPSim2 is not importable.\n"
            "  conda install -c conda-forge fpsim2\n"
            f"  (import failed with: {exc})"
        )

    missing_sidecars = [h5.name for h5, sc in shards if sc is None]
    if missing_sidecars and not args.no_smiles and rank == 0:
        log.warning(
            "No sidecar index for %d shard(s) (%s…) — those hits will carry "
            "ids and coefficients but no SMILES.",
            len(missing_sidecars), missing_sidecars[0],
        )

    my_shards = shards[rank::size] if size > 1 else shards
    if rank == 0:
        log.info("%d query/ies over %d shard(s)%s", len(queries), len(shards),
                 f" across {size} rank(s)" if size > 1 else "")

    t0 = time.time()
    rows: List[Tuple[str, int, str, float]] = []
    for h5, sidecar in my_shards:
        rows.extend(search_shard(h5, sidecar, queries, args))

    # Gather to rank 0. Hit lists can be large at low thresholds; --top-n
    # trims per shard before this point to keep the transfer bounded.
    if comm is not None:
        gathered = comm.gather(rows, root=0)
        if rank != 0:
            return
        rows = [r for part in gathered for r in part]

    merged = merge_hits(rows, args.top_n)

    out_path = Path(args.output)
    total = 0
    unresolved = 0
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["query_name", "query_smiles", "zinc_id", "smiles", "coeff"]
        )
        query_smiles = dict(queries)
        for name, _ in queries:
            for mol_id, smiles, coeff in merged.get(name, []):
                writer.writerow([name, query_smiles[name], mol_id, smiles,
                                 f"{coeff:.4f}"])
                total += 1
                if not smiles:
                    unresolved += 1

    log.info("Searched in %.1f s", time.time() - t0)
    for name, _ in queries:
        log.info("  %-16s %8s hit(s) >= %.2f", name,
                 f"{len(merged.get(name, [])):,}", args.threshold)
    log.info("Wrote %s hit(s) to %s", f"{total:,}", out_path)

    if unresolved and not args.no_smiles:
        log.warning(
            "%s hit(s) had no SMILES in any sidecar. If the build used "
            "--allow-large-ids this is the uint32 truncation described in "
            "fpsim2_build_db.py.", f"{unresolved:,}",
        )
    if total == 0:
        log.info("No hits — try a lower --threshold.")


if __name__ == "__main__":
    main()
