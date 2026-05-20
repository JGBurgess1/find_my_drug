"""
Convert top-hit SMILES to 3D PDBQT files ready for AutoDock Vina.

Reads the CSV produced by 01_extract_top_hits.py, embeds each molecule in 3D
using RDKit ETKDGv3 + MMFF94s, then writes a PDBQT via Open Babel (obabel)
with Gasteiger partial charges.

PDBQT files are organised into subdirectories of --subdir-size files each to
avoid filesystem bottlenecks with 1M files in a single directory.

Usage:
    python 02_smiles_to_pdbqt.py \
        --input      top_1M_hits.csv \
        --outdir     Vina_Ona_Repo/top_1M_pdbqt \
        --workers    32 \
        --subdir-size 10000
"""

import argparse
import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

MAX_MW        = 800    # Da  — skip unusually heavy molecules
MAX_ROTATABLE = 15     # bonds — very flexible molecules dock poorly
FF_STEPS      = 500    # MMFF94s minimisation steps


# ── Must be top-level for ProcessPoolExecutor pickling ───────────────────────

def _standardise(mol: Chem.Mol):
    try:
        clean   = rdMolStandardize.Cleanup(mol)
        largest = rdMolStandardize.LargestFragmentChooser().choose(clean)
        return  rdMolStandardize.Uncharger().uncharge(largest)
    except Exception:
        return None


def embed_molecule(zinc_id: str, smi: str, subdir: str) -> dict:
    """
    Embed one SMILES into 3D, minimise, write PDBQT via obabel.
    Returns a result dict with keys: zinc_id, smiles, status, path.
    """
    result = {"zinc_id": zinc_id, "smiles": smi, "status": "ok", "path": None}
    outdir = Path(subdir)

    pdbqt_path = outdir / f"{zinc_id}.pdbqt"
    if pdbqt_path.exists():
        result.update(path=str(pdbqt_path), status="already_exists")
        return result

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        result.update(status="invalid_smiles")
        return result

    if Descriptors.ExactMolWt(mol) > MAX_MW:
        result.update(status="too_heavy")
        return result

    if AllChem.CalcNumRotatableBonds(mol) > MAX_ROTATABLE:
        result.update(status="too_flexible")
        return result

    mol = _standardise(mol)
    if mol is None:
        result.update(status="standardise_failed")
        return result

    mol = Chem.AddHs(mol)
    mol.SetProp("_Name", zinc_id)

    params            = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)
    if not cids:
        result.update(status="embed_failed")
        return result

    AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94s", maxIters=FF_STEPS)

    sdf_path = outdir / f"{zinc_id}.sdf"
    writer   = Chem.SDWriter(str(sdf_path))
    writer.write(mol, confId=int(cids[0]))
    writer.close()

    try:
        proc = subprocess.run(
            ["obabel", str(sdf_path), "-O", str(pdbqt_path),
             "--partialcharge", "gasteiger", "-h"],
            capture_output=True,
            timeout=60,
        )
        if proc.returncode == 0 and pdbqt_path.exists():
            sdf_path.unlink(missing_ok=True)
            result.update(path=str(pdbqt_path))
        else:
            result.update(status="obabel_failed", path=str(sdf_path))
    except FileNotFoundError:
        result.update(status="obabel_not_found", path=str(sdf_path))
    except subprocess.TimeoutExpired:
        result.update(status="obabel_timeout", path=str(sdf_path))

    return result


def process_batch(args: tuple) -> list:
    """Worker entry point: (batch of (zinc_id, smi), subdir_str)"""
    batch, subdir = args
    return [embed_molecule(zinc_id, smi, subdir) for zinc_id, smi in batch]


# ── Helpers ───────────────────────────────────────────────────────────────────

def subdir_for_index(base: Path, idx: int, subdir_size: int) -> Path:
    """Return e.g. base/0000000-0009999/ for idx 0..9999."""
    start = (idx // subdir_size) * subdir_size
    end   = start + subdir_size - 1
    return base / f"{start:07d}-{end:07d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert top-hit SMILES to 3D PDBQT files for AutoDock Vina"
    )
    parser.add_argument("--input",        "-i", required=True,
                        help="CSV from 01_extract_top_hits.py")
    parser.add_argument("--outdir",       "-o", default="Vina_Ona_Repo/top_1M_pdbqt",
                        help="Root output directory for PDBQT files")
    parser.add_argument("--workers",      "-w", type=int, default=os.cpu_count(),
                        help="Parallel worker processes (default: all CPUs)")
    parser.add_argument("--batch-size",   "-b", type=int, default=5000,
                        help="Compounds per worker task (default: 5000)")
    parser.add_argument("--subdir-size",  "-s", type=int, default=10_000,
                        help="Max PDBQT files per subdirectory (default: 10,000)")
    parser.add_argument("--zinc-col",           default="zinc_id")
    parser.add_argument("--smiles-col",         default="smiles")
    args = parser.parse_args()

    base_dir = Path(args.outdir)
    base_dir.mkdir(parents=True, exist_ok=True)

    df    = pd.read_csv(args.input, usecols=[args.zinc_col, args.smiles_col])
    df    = df.dropna().reset_index(drop=True)
    pairs = list(zip(df[args.zinc_col].astype(str), df[args.smiles_col]))
    log.info(f"Loaded {len(pairs):,} compounds from {args.input}")

    # Pre-create all subdirectories
    unique_subdirs = {
        subdir_for_index(base_dir, i, args.subdir_size)
        for i in range(len(pairs))
    }
    for d in unique_subdirs:
        d.mkdir(parents=True, exist_ok=True)
    log.info(f"Created {len(unique_subdirs)} subdirectories under {base_dir}")

    # Build worker tasks: each batch gets a (possibly mixed) subdir assignment;
    # assign each compound to its subdir individually via the batch index offset
    def make_tasks():
        for batch_start in range(0, len(pairs), args.batch_size):
            batch = []
            for j, (zinc_id, smi) in enumerate(
                pairs[batch_start : batch_start + args.batch_size]
            ):
                subdir = subdir_for_index(base_dir, batch_start + j, args.subdir_size)
                batch.append((zinc_id, smi, str(subdir)))
            yield batch

    # Flatten: pass (zinc_id, smi, subdir) triples, but process_batch expects
    # (zinc_id, smi) + a shared subdir. Since files in one batch may span
    # multiple subdirs, process them individually instead of by shared subdir.
    def process_triple_batch(triples: list) -> list:
        return [embed_molecule(z, s, d) for z, s, d in triples]

    all_tasks = list(make_tasks())
    log.info(f"Dispatching {len(all_tasks)} batches to {args.workers} workers")

    results   = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_triple_batch, task): i
                   for i, task in enumerate(all_tasks)}
        for fut in as_completed(futures):
            batch_results = fut.result()
            results.extend(batch_results)
            completed += len(batch_results)
            if completed % 50_000 == 0 or completed == len(pairs):
                ok = sum(1 for r in results if "ok" in r["status"])
                log.info(f"  {completed:>9,}/{len(pairs):,}  ok: {ok:,}")

    ok_results     = [r for r in results if "ok" in r["status"]]
    failed_results = [r for r in results if "ok" not in r["status"]]
    log.info(f"\nSuccess: {len(ok_results):,}   Failed: {len(failed_results):,}")

    if failed_results:
        fail_path = base_dir / "failed_molecules.csv"
        pd.DataFrame(failed_results).to_csv(fail_path, index=False)
        log.warning(f"Failed molecules written to {fail_path}")

    manifest_path = base_dir / "manifest.csv"
    pd.DataFrame(ok_results)[["zinc_id", "smiles", "path"]].to_csv(
        manifest_path, index=False
    )
    log.info(f"Manifest: {manifest_path}")
    log.info(f"PDBQT files ready in: {base_dir}")
    log.info(
        f"\nNext step:\n"
        f"  mpiexec -n <N> python Vina_Ona_Repo/run_docking.py \\\n"
        f"      --config <config.yaml> \\\n"
        f"      --ligands {base_dir} --recursive \\\n"
        f"      --output docking_results_round2.csv"
    )


if __name__ == "__main__":
    main()
