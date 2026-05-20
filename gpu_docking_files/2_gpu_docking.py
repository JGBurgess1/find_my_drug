"""
Step 2: GPU-Accelerated Molecular Docking
──────────────────────────────────────────
Supports four docking backends, selectable via --backend:

  unidock    →  Uni-Dock v1.1+  (CUDA, DP Technology)
                https://github.com/dptech-corp/Uni-Dock
                Docks a BATCH of PDBQT ligands in one GPU kernel call via
                --gpu_batch.  For very large libraries a --ligand_index text
                file is used instead (avoids shell ARG_MAX limits).
                Also supports the higher-level `unidocktools unidock_pipeline`
                (set --use_unidock_tools) which handles SDF input natively.
                Typical throughput: 1,000×+ faster than AutoDock Vina CPU.

  vina-gpu   →  Vina-GPU 2.1   (CUDA, open-source fork of AutoDock Vina)
                https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1

  autodock   →  AutoDock-GPU   (CUDA/OpenCL, classical AutoDock4 force field)
                https://github.com/ccsb-scripps/AutoDock-GPU

  gnina      →  gnina           (CUDA, CNN-based scoring, Vina-compatible)
                https://github.com/gnina/gnina

Architecture notes
──────────────────
UniDock (raw binary):
  • --gpu_batch   accepts PDBQT files (space-separated), up to ARG_MAX.
  • --ligand_index accepts a text file listing PDBQT paths, one per line;
    used automatically when batch size > 500 to avoid shell limits.
  • Output is one <stem>_out.pdbqt per ligand written to --dir.
  • Scores are in REMARK VINA RESULT lines of each output_old PDBQT.

UniDock Tools (unidocktools):
  • Wraps the binary; accepts SDF input directly.
  • unidocktools ligandprep   — convert SDF → torsion-tree PDBQT.
  • unidocktools unidock_pipeline — end-to-end receptor+ligand → scored SDF.
  • install: pip install git+https://github.com/dptech-corp/Uni-Dock.git
                         #subdirectory=unidock_tools

All other backends (vina-gpu, autodock, gnina) process ligands one-at-a-time
parallelised via a thread pool with a GPU token queue.

Usage — UniDock (raw binary, recommended):
    python 2_gpu_docking.py \\
        --manifest   ligands_3d/manifest.txt \\
        --receptor   protein.pdbqt \\
        --config     docking_box.conf \\
        --outdir     docking_results \\
        --backend    unidock \\
        --gpus       0,1 \\
        --batch_size 1200 \\
        --search_mode balance \\
        --output_csv results.csv

Usage — UniDock Tools (SDF input, no pre-prep needed):
    python 2_gpu_docking.py \\
        --manifest   ligands_3d/manifest.txt \\
        --receptor   protein.pdbqt \\
        --config     docking_box.conf \\
        --outdir     docking_results \\
        --backend    unidock \\
        --use_unidock_tools \\
        --gpus       0 \\
        --batch_size 1200 \\
        --output_csv results.csv

Usage — Vina-GPU / gnina / AutoDock-GPU (single-ligand path):
    python 2_gpu_docking.py \\
        --manifest  ligands_3d/manifest.txt \\
        --receptor  protein.pdbqt \\
        --config    docking_box.conf \\
        --outdir    docking_results \\
        --backend   vina-gpu \\
        --gpus      0,1 \\
        --workers   4 \\
        --output_csv results.csv

docking_box.conf example:
    center_x = 12.5
    center_y = -3.2
    center_z = 44.1
    size_x   = 20
    size_y   = 20
    size_z   = 20
    exhaustiveness = 8
    num_modes = 5

Receptor must be pre-prepared (protonated, Gasteiger charges assigned):
    prepare_receptor -r protein.pdb -o protein.pdbqt -A hydrogens
"""

import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Backend discovery ─────────────────────────────────────────────────────────
BACKENDS = {
    "unidock":  ["unidock"],
    "vina-gpu": ["Vina-GPU-2-1", "vina_gpu", "AutoDock-Vina-GPU"],
    "autodock": ["AutoDock-GPU", "autodock_gpu_128wi", "autodock_gpu_64wi"],
    "gnina":    ["gnina"],
}
UNIDOCK_TOOLS_EXE = "unidocktools"
UNIDOCK_SEARCH_MODES = ("fast", "balance", "detail")

# If a batch has more than this many ligands, write a --ligand_index file
# instead of passing all paths on the command line (avoids ARG_MAX limits).
LIGAND_INDEX_THRESHOLD = 500


def find_executable(backend: str) -> str:
    for candidate in BACKENDS.get(backend, [backend]):
        path = shutil.which(candidate)
        if path:
            return path
    raise FileNotFoundError(
        f"No executable found for backend '{backend}'.\n"
        f"Tried: {BACKENDS.get(backend, [backend])}\n"
        f"Ensure it is compiled/installed and on your PATH.\n"
        f"  conda install -c https://conda.mlops.dp.tech/caic -c conda-forge unidock"
    )


# ── Config parsing ────────────────────────────────────────────────────────────
def parse_conf(conf_path: str) -> dict:
    cfg = {}
    with open(conf_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


# ── Score parsers ─────────────────────────────────────────────────────────────
def parse_vina_score(output_text: str) -> float | None:
    """
    Vina / Vina-GPU / UniDock stdout table:
        1     -8.345      0.000      0.000
    Also handles REMARK lines inside PDBQT/SDF output_old files.
    """
    for line in output_text.splitlines():
        m = re.match(r"\s+1\s+([-\d.]+)", line)
        if m:
            return float(m.group(1))
    m = re.search(r"REMARK VINA RESULT:\s+([-\d.]+)", output_text)
    if m:
        return float(m.group(1))
    return None


def parse_autodock_gpu_score(output_text: str) -> float | None:
    m = re.search(r"RANKING\s+1\s+\S+\s+\S+\s+([-\d.]+)", output_text)
    if m:
        return float(m.group(1))
    m = re.search(r"Estimated Free Energy of Binding\s*=\s*([-\d.]+)", output_text)
    if m:
        return float(m.group(1))
    return None


def parse_gnina_score(output_text: str) -> float | None:
    return parse_vina_score(output_text)


def parse_unidock_output_pdbqt(pdbqt_path: Path) -> float | None:
    """
    UniDock writes per-ligand output_old PDBQT to --dir/<stem>_out.pdbqt.
    The best-pose score is on a REMARK VINA RESULT line:
        REMARK VINA RESULT:    -8.345      0.000      0.000
    """
    if not pdbqt_path.exists():
        return None
    text = pdbqt_path.read_text(errors="replace")
    m = re.search(r"REMARK VINA RESULT:\s+([-\d.]+)", text)
    if m:
        return float(m.group(1))
    # Fallback: first numeric REMARK line
    m = re.search(r"REMARK\s+([-\d.]+)", text)
    if m:
        return float(m.group(1))
    return None


def parse_unidock_tools_sdf(sdf_path: Path) -> float | None:
    """
    unidocktools writes scored SDF files.  Score stored as:
        > <docking_score>
        -8.345
    or as a REMARK VINA RESULT line inside the SDF.
    """
    if not sdf_path.exists():
        return None
    text = sdf_path.read_text(errors="replace")
    m = re.search(r">\s*<docking_score>\s*\n\s*([-\d.]+)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"REMARK VINA RESULT:\s+([-\d.]+)", text)
    if m:
        return float(m.group(1))
    return None


SCORE_PARSERS = {
    "vina-gpu": parse_vina_score,
    "autodock": parse_autodock_gpu_score,
    "gnina":    parse_vina_score,
    # unidock reads per-ligand output_old files — handled in batch functions below
}


# ═══════════════════════════════════════════════════════════════════════════════
# UniDock — raw binary batch path
# ═══════════════════════════════════════════════════════════════════════════════
def dock_batch_unidock(
    batch: list[tuple[str, str, str]],   # [(zinc_id, smiles, pdbqt_path), ...]
    receptor: str,
    cfg: dict,
    exe: str,
    gpu_id: int,
    outdir: Path,
    search_mode: str = "balance",
    timeout: int = 600,
) -> list[dict]:
    """
    Submit one UniDock raw-binary batch call for `batch` ligands.

    UniDock takes PDBQT files via --gpu_batch (or --ligand_index for large
    batches) and writes <stem>_out.pdbqt into --dir for each ligand.
    Scores are parsed from REMARK VINA RESULT lines in those output_old files.
    """
    batch_tag    = f"gpu{gpu_id}_{os.getpid()}_{int(time.time() * 1000)}"
    batch_outdir = outdir / batch_tag
    batch_outdir.mkdir(parents=True, exist_ok=True)

    # Separate valid (have a PDBQT) from invalid entries
    valid:   list[tuple[str, str, Path]] = []
    results: list[dict] = []

    for zinc_id, smiles, lig_path in batch:
        p = Path(lig_path)
        if p.suffix == ".pdbqt" and p.exists():
            valid.append((zinc_id, smiles, p))
        else:
            # Try to find a .pdbqt sibling if a .pdb was stored
            pdbqt = p.with_suffix(".pdbqt")
            if pdbqt.exists():
                valid.append((zinc_id, smiles, pdbqt))
            else:
                results.append({
                    "zinc_id": zinc_id, "smiles": smiles,
                    "docking_score": None, "status": "no_pdbqt_found",
                })

    if not valid:
        return results

    pdbqt_paths = [str(v[2]) for v in valid]

    # Build the command
    # For large batches, write a --ligand_index file to stay under ARG_MAX
    if len(pdbqt_paths) > LIGAND_INDEX_THRESHOLD:
        index_file = batch_outdir / "ligand_index.txt"
        index_file.write_text("\n".join(pdbqt_paths))
        ligand_args = ["--ligand_index", str(index_file)]
    else:
        ligand_args = ["--gpu_batch", *pdbqt_paths]

    cmd = [
        exe,
        "--receptor",    receptor,
        *ligand_args,
        "--center_x",    cfg.get("center_x", "0"),
        "--center_y",    cfg.get("center_y", "0"),
        "--center_z",    cfg.get("center_z", "0"),
        "--size_x",      cfg.get("size_x", "20"),
        "--size_y",      cfg.get("size_y", "20"),
        "--size_z",      cfg.get("size_z", "20"),
        "--num_modes",   cfg.get("num_modes", "5"),
        "--search_mode", search_mode,
        "--scoring",     cfg.get("scoring", "vina"),
        "--dir",         str(batch_outdir),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_file = batch_outdir / "unidock.log"
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
        log_file.write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            log.warning(
                f"[UniDock] batch {batch_tag} exited {proc.returncode} — "
                f"see {log_file}"
            )
    except subprocess.TimeoutExpired:
        return results + [
            {"zinc_id": z, "smiles": s, "docking_score": None, "status": "timeout"}
            for z, s, _ in valid
        ]
    except Exception as exc:
        return results + [
            {"zinc_id": z, "smiles": s, "docking_score": None,
             "status": f"error:{exc}"}
            for z, s, _ in valid
        ]

    # Collect per-ligand scores from output_old PDBQTs
    for zinc_id, smiles, pdbqt_in in valid:
        out_pdbqt = batch_outdir / f"{pdbqt_in.stem}_out.pdbqt"
        score     = parse_unidock_output_pdbqt(out_pdbqt)
        if score is None:
            results.append({
                "zinc_id": zinc_id, "smiles": smiles,
                "docking_score": None, "status": "parse_failed",
            })
        else:
            results.append({
                "zinc_id": zinc_id, "smiles": smiles,
                "docking_score": score, "status": "ok",
            })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# UniDock Tools — high-level Python pipeline path (SDF input)
# ═══════════════════════════════════════════════════════════════════════════════
def dock_batch_unidock_tools(
    batch: list[tuple[str, str, str]],
    receptor: str,
    cfg: dict,
    gpu_id: int,
    outdir: Path,
    search_mode: str = "balance",
    timeout: int = 600,
) -> list[dict]:
    """
    Use `unidocktools unidock_pipeline` to dock a batch of ligands.
    This path accepts SDF files directly and handles internal ligand prep.

    unidocktools unidock_pipeline:
      -r  receptor PDBQT
      -i  text file listing SDF paths (one per line)
      -sd save directory
      -cx/-cy/-cz  box centre
      -sx/-sy/-sz  box dimensions
      -sf  scoring function (vina/vinardo/ad4)
    """
    exe = shutil.which(UNIDOCK_TOOLS_EXE)
    if exe is None:
        raise FileNotFoundError(
            f"'{UNIDOCK_TOOLS_EXE}' not found. Install with:\n"
            "  pip install 'git+https://github.com/dptech-corp/Uni-Dock.git"
            "#subdirectory=unidock_tools'"
        )

    batch_tag    = f"gpu{gpu_id}_{os.getpid()}_{int(time.time() * 1000)}"
    batch_outdir = outdir / batch_tag
    batch_outdir.mkdir(parents=True, exist_ok=True)

    # Resolve SDF paths (unidock_tools accepts SDF natively)
    valid:   list[tuple[str, str, Path]] = []
    results: list[dict] = []

    for zinc_id, smiles, lig_path in batch:
        p   = Path(lig_path)
        sdf = p.with_suffix(".sdf")   # step 1 always writes .sdf alongside .pdbqt
        if sdf.exists():
            valid.append((zinc_id, smiles, sdf))
        elif p.suffix == ".sdf" and p.exists():
            valid.append((zinc_id, smiles, p))
        else:
            results.append({
                "zinc_id": zinc_id, "smiles": smiles,
                "docking_score": None, "status": "no_sdf_found",
            })

    if not valid:
        return results

    # Write ligand index file
    index_file = batch_outdir / "ligand_index.txt"
    index_file.write_text("\n".join(str(v[2]) for v in valid))

    cmd = [
        exe, "unidock_pipeline",
        "-r",  receptor,
        "-i",  str(index_file),
        "-sd", str(batch_outdir),
        "-cx", cfg.get("center_x", "0"),
        "-cy", cfg.get("center_y", "0"),
        "-cz", cfg.get("center_z", "0"),
        "-sx", cfg.get("size_x", "20"),
        "-sy", cfg.get("size_y", "20"),
        "-sz", cfg.get("size_z", "20"),
        "-sf", cfg.get("scoring", "vina"),
        "-nm", cfg.get("num_modes", "5"),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_file = batch_outdir / "unidocktools.log"
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
        log_file.write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            log.warning(
                f"[unidocktools] batch {batch_tag} exited {proc.returncode} — "
                f"see {log_file}"
            )
    except subprocess.TimeoutExpired:
        return results + [
            {"zinc_id": z, "smiles": s, "docking_score": None, "status": "timeout"}
            for z, s, _ in valid
        ]
    except Exception as exc:
        return results + [
            {"zinc_id": z, "smiles": s, "docking_score": None,
             "status": f"error:{exc}"}
            for z, s, _ in valid
        ]

    # unidock_pipeline writes <stem>_out.sdf (scored) to batch_outdir
    for zinc_id, smiles, sdf_in in valid:
        out_sdf = batch_outdir / f"{sdf_in.stem}_out.sdf"
        score   = parse_unidock_tools_sdf(out_sdf)
        if score is None:
            results.append({
                "zinc_id": zinc_id, "smiles": smiles,
                "docking_score": None, "status": "parse_failed",
            })
        else:
            results.append({
                "zinc_id": zinc_id, "smiles": smiles,
                "docking_score": score, "status": "ok",
            })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Single-ligand docking — Vina-GPU / AutoDock-GPU / gnina
# ═══════════════════════════════════════════════════════════════════════════════
def dock_one(
    zinc_id: str,
    smiles: str,
    ligand_path: str,
    receptor: str,
    cfg: dict,
    backend: str,
    exe: str,
    gpu_id: int,
    outdir: Path,
    timeout: int = 120,
) -> dict:
    """Run docking for a single ligand; return a result dict."""
    out_pdbqt = outdir / f"{zinc_id}_out.pdbqt"
    log_file  = outdir / f"{zinc_id}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if backend in ("vina-gpu", "gnina"):
        cmd = [
            exe,
            "--receptor",       receptor,
            "--ligand",         ligand_path,
            "--out",            str(out_pdbqt),
            "--center_x",       cfg.get("center_x", "0"),
            "--center_y",       cfg.get("center_y", "0"),
            "--center_z",       cfg.get("center_z", "0"),
            "--size_x",         cfg.get("size_x", "20"),
            "--size_y",         cfg.get("size_y", "20"),
            "--size_z",         cfg.get("size_z", "20"),
            "--exhaustiveness", cfg.get("exhaustiveness", "8"),
            "--num_modes",      cfg.get("num_modes", "5"),
        ]
        if backend == "vina-gpu":
            cmd += ["--thread", "8000"]
        elif backend == "gnina":
            cmd += ["--no_gpu", "false", "--gpu_device", str(gpu_id)]

    elif backend == "autodock":
        _write_autodock_dpf(zinc_id, ligand_path, receptor, cfg, outdir)
        cmd = [
            exe,
            "--ffile", receptor.replace(".pdbqt", ".maps.fld"),
            "--lfile", ligand_path,
            "--nrun",  "10",
        ]
        env["GPU_DEVICE"] = str(gpu_id)

    else:
        return {"zinc_id": zinc_id, "smiles": smiles,
                "docking_score": None, "status": f"unknown_backend:{backend}"}

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
        combined = proc.stdout + proc.stderr
        log_file.write_text(combined)
        score = SCORE_PARSERS[backend](combined)
        if score is None:
            return {"zinc_id": zinc_id, "smiles": smiles,
                    "docking_score": None, "status": "parse_failed"}
        return {"zinc_id": zinc_id, "smiles": smiles,
                "docking_score": score, "status": "ok"}
    except subprocess.TimeoutExpired:
        return {"zinc_id": zinc_id, "smiles": smiles,
                "docking_score": None, "status": "timeout"}
    except Exception as exc:
        return {"zinc_id": zinc_id, "smiles": smiles,
                "docking_score": None, "status": f"error:{exc}"}


def _write_autodock_dpf(zinc_id, ligand_path, receptor, cfg, outdir) -> str:
    dpf_path = outdir / f"{zinc_id}.dpf"
    with open(dpf_path, "w") as fh:
        fh.write("ligand_types A C HD N NA OA SA\n")
        fh.write(f"fld {Path(receptor).stem}.maps.fld\n")
        fh.write(f"move {ligand_path}\n")
        fh.write("nruns 10\nga_pop_size 150\nga_num_evals 2500000\nanalysis\n")
    return str(dpf_path)


# ── GPU token queue ───────────────────────────────────────────────────────────
class GPUPool:
    def __init__(self, gpu_ids: list[int]):
        self._q: Queue = Queue()
        for g in gpu_ids:
            self._q.put(g)

    def acquire(self) -> int:
        return self._q.get()

    def release(self, gpu_id: int):
        self._q.put(gpu_id)


def _log_progress(done: int, total: int, ok: int, t0: float):
    elapsed = time.time() - t0
    rate    = done / elapsed if elapsed > 0 else 0
    eta     = (total - done) / rate if rate > 0 else 0
    log.info(
        f"  {done}/{total}  ✓{ok}  {rate:.1f} cpd/s  ETA {eta/60:.1f} min"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated molecular docking")
    parser.add_argument("--manifest",         required=True,
                        help="manifest.txt from step 1 (zinc_id TAB smiles TAB path)")
    parser.add_argument("--receptor",         required=True, help="Receptor .pdbqt")
    parser.add_argument("--config",           required=True, help="Docking box .conf")
    parser.add_argument("--outdir",           default="docking_results")
    parser.add_argument("--backend",          default="unidock",
                        choices=["unidock", "vina-gpu", "autodock", "gnina"])
    parser.add_argument("--gpus",             default="0",
                        help="Comma-separated GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--workers",          type=int, default=2,
                        help="Concurrent batch jobs (UniDock) or "
                             "concurrent per-ligand processes (other backends)")
    # ── UniDock-specific ──────────────────────────────────────────────────────
    parser.add_argument("--batch_size",       type=int, default=1200,
                        help="[UniDock] Ligands per GPU batch. "
                             "Reduce if CUDA OOM. (default: 1200)")
    parser.add_argument("--search_mode",      default="balance",
                        choices=list(UNIDOCK_SEARCH_MODES),
                        help="[UniDock] fast / balance / detail (default: balance)")
    parser.add_argument("--use_unidock_tools", action="store_true",
                        help="[UniDock] Use the high-level `unidocktools` CLI "
                             "instead of the raw binary. Accepts SDF input "
                             "and handles ligand prep internally.")
    # ── General ───────────────────────────────────────────────────────────────
    parser.add_argument("--timeout",          type=int, default=600,
                        help="Subprocess timeout in seconds (default: 600)")
    parser.add_argument("--output_csv",       default="docking_results.csv")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gpu_ids  = [int(g.strip()) for g in args.gpus.split(",")]
    gpu_pool = GPUPool(gpu_ids)
    cfg      = parse_conf(args.config)

    # Validate executable (skip for unidock_tools — checked inside the function)
    if not (args.backend == "unidock" and args.use_unidock_tools):
        try:
            exe = find_executable(args.backend)
            log.info(f"Backend : {args.backend}  →  {exe}")
        except FileNotFoundError as e:
            log.error(str(e))
            sys.exit(1)
    else:
        exe = ""  # unused
        log.info("Backend : unidock (via unidocktools pipeline)")

    # ── Load manifest ──────────────────────────────────────────────────────────
    entries: list[tuple[str, str, str]] = []
    with open(args.manifest) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                entries.append((parts[0], parts[1], parts[2]))

    log.info(f"Ligands : {len(entries)}")
    log.info(f"GPUs    : {gpu_ids}")

    results: list[dict] = []
    t0 = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # UniDock — batch path (raw binary or unidock_tools)
    # ══════════════════════════════════════════════════════════════════════════
    if args.backend == "unidock":
        batches = [
            entries[i : i + args.batch_size]
            for i in range(0, len(entries), args.batch_size)
        ]
        log.info(
            f"Batches : {len(batches)} × up to {args.batch_size}  "
            f"search_mode={args.search_mode}"
        )

        max_workers = args.workers * len(gpu_ids)
        done_count  = 0

        def _unidock_worker(batch):
            gpu_id = gpu_pool.acquire()
            try:
                if args.use_unidock_tools:
                    return dock_batch_unidock_tools(
                        batch=batch, receptor=args.receptor, cfg=cfg,
                        gpu_id=gpu_id, outdir=outdir,
                        search_mode=args.search_mode, timeout=args.timeout,
                    )
                else:
                    return dock_batch_unidock(
                        batch=batch, receptor=args.receptor, cfg=cfg,
                        exe=exe, gpu_id=gpu_id, outdir=outdir,
                        search_mode=args.search_mode, timeout=args.timeout,
                    )
            finally:
                gpu_pool.release(gpu_id)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_unidock_worker, b): i
                       for i, b in enumerate(batches)}
            for fut in as_completed(futures):
                batch_res = fut.result()
                results.extend(batch_res)
                done_count += len(batch_res)
                ok = sum(1 for r in results if r["status"] == "ok")
                _log_progress(done_count, len(entries), ok, t0)

    # ══════════════════════════════════════════════════════════════════════════
    # Vina-GPU / AutoDock-GPU / gnina — single-ligand path
    # ══════════════════════════════════════════════════════════════════════════
    else:
        max_workers = args.workers * len(gpu_ids)
        log.info(f"Workers : {max_workers} concurrent")

        def _single_worker(entry):
            zinc_id, smiles, lig_path = entry
            gpu_id = gpu_pool.acquire()
            try:
                return dock_one(
                    zinc_id=zinc_id, smiles=smiles, ligand_path=lig_path,
                    receptor=args.receptor, cfg=cfg, backend=args.backend,
                    exe=exe, gpu_id=gpu_id, outdir=outdir, timeout=args.timeout,
                )
            finally:
                gpu_pool.release(gpu_id)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_single_worker, e): e[0] for e in entries}
            for i, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if i % 100 == 0 or i == len(entries):
                    ok = sum(1 for r in results if r["status"] == "ok")
                    _log_progress(i, len(entries), ok, t0)

    # ── Write CSV ──────────────────────────────────────────────────────────────
    csv_path = Path(args.output_csv)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["zinc_id", "smiles", "docking_score", "status"]
        )
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["docking_score"] or 0):
            writer.writerow(r)

    ok      = sum(1 for r in results if r["status"] == "ok")
    failed  = len(results) - ok
    elapsed = time.time() - t0
    log.info(
        f"\nDocking complete.  ✓ {ok}  ✗ {failed}  "
        f"Time: {elapsed/60:.1f} min  →  {csv_path}"
    )


if __name__ == "__main__":
    main()
