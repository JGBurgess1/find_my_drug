"""
Step 3: End-to-End Pipeline
────────────────────────────
Orchestrates the full workflow:
  SMILES file  →  3D PDBQT  →  GPU Docking  →  CSV

Output CSV columns:
  zinc_id | smiles | docking_score

Usage — UniDock (recommended):
    python 3_pipeline.py \\
        --input     compounds.smi \\
        --receptor  protein.pdbqt \\
        --config    box.conf \\
        --backend   unidock \\
        --gpus      0,1 \\
        --workers   4 \\
        --batch_size 1200 \\
        --search_mode balance \\
        --output_old    results.csv

Usage — unidock_tools (SDF input, no manual ligand prep needed):
    python 3_pipeline.py \\
        --input     compounds.smi \\
        --receptor  protein.pdbqt \\
        --config    box.conf \\
        --backend   unidock \\
        --use_unidock_tools \\
        --gpus      0 \\
        --output_old    results.csv

Usage — Vina-GPU / gnina / AutoDock-GPU:
    python 3_pipeline.py \\
        --input     compounds.smi \\
        --receptor  protein.pdbqt \\
        --config    box.conf \\
        --backend   vina-gpu \\
        --gpus      0,1 \\
        --workers   8 \\
        --output_old    results.csv

For large libraries (>100k compounds), run steps 1 and 2 separately so
you can checkpoint and resume without re-doing the 3D embedding.
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

STEP1 = Path(__file__).parent / "1_smiles_to_3d.py"
STEP2 = Path(__file__).parent / "2_gpu_docking.py"


def run(cmd: list, label: str):
    log.info(f"▶ {label}")
    log.info(f"  CMD: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log.error(f"  {label} FAILED (exit {proc.returncode})")
        sys.exit(proc.returncode)
    log.info(f"  {label} done in {elapsed/60:.1f} min")


def finalise_csv(raw_csv: Path, output_csv: Path):
    """
    Read the raw docking CSV, keep only successful rows,
    drop the internal 'status' column, sort by score ascending,
    and write the clean 3-column output_old.
    """
    rows = []
    with open(raw_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("status") == "ok" and row.get("docking_score"):
                rows.append({
                    "zinc_id":       row["zinc_id"],
                    "smiles":        row["smiles"],
                    "docking_score": float(row["docking_score"]),
                })

    rows.sort(key=lambda r: r["docking_score"])

    with open(output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["zinc_id", "smiles", "docking_score"]
        )
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"\n{'─'*60}")
    log.info(f"  Final CSV  : {output_csv}")
    log.info(f"  Scored     : {len(rows)} compounds")
    if rows:
        log.info(f"  Best score : {rows[0]['docking_score']:.3f} kcal/mol"
                 f"  ({rows[0]['zinc_id']})")
        log.info(f"  Median     : {rows[len(rows)//2]['docking_score']:.3f} kcal/mol")
    log.info(f"{'─'*60}")
    log.info("\nTop 10 hits:")
    log.info(f"{'ZINC ID':<22} {'Score':>8}  SMILES")
    log.info("─" * 80)
    for r in rows[:10]:
        smi_clip = r["smiles"][:50] + ("…" if len(r["smiles"]) > 50 else "")
        log.info(f"{r['zinc_id']:<22} {r['docking_score']:>8.3f}  {smi_clip}")


def main():
    parser = argparse.ArgumentParser(description="Full SMILES→Dock→CSV pipeline")

    # Inputs
    parser.add_argument("--input",            required=True, help="Input .smi file")
    parser.add_argument("--receptor",         required=True, help="Receptor .pdbqt")
    parser.add_argument("--config",           required=True, help="Docking box .conf")

    # Backend
    parser.add_argument("--backend",          default="unidock",
                        choices=["unidock", "vina-gpu", "autodock", "gnina"])
    parser.add_argument("--gpus",             default="0",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--workers",          type=int, default=4,
                        help="Parallel workers (batches for UniDock, "
                             "ligands for other backends)")
    parser.add_argument("--timeout",          type=int, default=600,
                        help="Per-batch / per-ligand timeout (s)")

    # UniDock-specific
    parser.add_argument("--batch_size",       type=int, default=1200,
                        help="[UniDock] Ligands per GPU batch (default: 1200)")
    parser.add_argument("--search_mode",      default="balance",
                        choices=["fast", "balance", "detail"],
                        help="[UniDock] Search mode (default: balance)")
    parser.add_argument("--use_unidock_tools", action="store_true",
                        help="[UniDock] Use unidocktools pipeline (SDF input, "
                             "handles ligand prep internally)")

    # Outputs / control
    parser.add_argument("--workdir",          default="pipeline_work",
                        help="Scratch directory for intermediates")
    parser.add_argument("--output_old",           default="results.csv",
                        help="Final output_old CSV (zinc_id, smiles, docking_score)")
    parser.add_argument("--skip_embed",       action="store_true",
                        help="Skip 3D embedding (reuse existing manifest)")

    args = parser.parse_args()

    workdir  = Path(args.workdir)
    ligdir   = workdir / "ligands_3d"
    dockdir  = workdir / "docking"
    raw_csv  = workdir / "raw_docking.csv"
    manifest = ligdir  / "manifest.txt"

    workdir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: SMILES → 3D PDBQT ──────────────────────────────────────────
    if not args.skip_embed:
        run(
            [sys.executable, str(STEP1),
             "--input",   args.input,
             "--outdir",  str(ligdir),
             "--workers", str(args.workers)],
            "Step 1: SMILES → 3D conformers",
        )
    else:
        log.info("Step 1 skipped (--skip_embed); using existing manifest")
        if not manifest.exists():
            log.error(f"Manifest not found: {manifest}")
            sys.exit(1)

    # ── Step 2: GPU Docking ─────────────────────────────────────────────────
    step2_cmd = [
        sys.executable, str(STEP2),
        "--manifest",    str(manifest),
        "--receptor",    args.receptor,
        "--config",      args.config,
        "--outdir",      str(dockdir),
        "--backend",     args.backend,
        "--gpus",        args.gpus,
        "--workers",     str(args.workers),
        "--timeout",     str(args.timeout),
        "--output_csv",  str(raw_csv),
    ]

    # Append UniDock-specific flags only when relevant
    if args.backend == "unidock":
        step2_cmd += [
            "--batch_size",  str(args.batch_size),
            "--search_mode", args.search_mode,
        ]
        if args.use_unidock_tools:
            step2_cmd.append("--use_unidock_tools")

    run(step2_cmd, "Step 2: GPU docking")

    # ── Step 3: Finalise CSV ────────────────────────────────────────────────
    finalise_csv(raw_csv, Path(args.output))


if __name__ == "__main__":
    main()
