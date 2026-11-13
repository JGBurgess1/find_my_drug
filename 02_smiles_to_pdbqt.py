"""
Convert top-hit SMILES to 3D PDBQT files ready for AutoDock Vina.

Reads either the CSV produced by 01_extract_top_hits.py or a whitespace-
delimited SMILES file of '<SMILES> <zinc_id>' lines, embeds each molecule in
3D using RDKit ETKDGv3 + MMFF94s, then converts to PDBQT via the Open Babel
Python API (openbabel / pybel) with Gasteiger partial charges.

The format is auto-detected from the file extension (.smi/.smiles/.ism/.txt
→ SMI, .csv/.tsv → CSV) and otherwise by sniffing the first line; override
with --format csv|smi.

No subprocess calls, no temp SDF files — RDKit writes the 3D SDF block
directly to a Python string, which is passed in-memory to Open Babel.

PDBQT files are organised into subdirectories of --subdir-size files each to
avoid filesystem bottlenecks with 1M files in a single directory.

Usage:
    python 02_smiles_to_pdbqt.py \
        --input      top_1M_hits.csv \
        --outdir     Vina_Ona_Repo/top_1M_pdbqt \
        --workers    32 \
        --subdir-size 10000

    python 02_smiles_to_pdbqt.py \
        --input      top_1M_smiles.smi \
        --outdir     Vina_Ona_Repo/top_1M_pdbqt \
        --workers    32
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# ── Open Babel Python API ─────────────────────────────────────────────────────
try:
    from openbabel import openbabel as ob
    from openbabel import pybel as pybel_ob
    OB_AVAILABLE = True
except ImportError:
    OB_AVAILABLE = False
    ob       = None
    pybel_ob = None

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


def _sdf_to_pdbqt(sdf_block: str, zinc_id: str, pdbqt_path: Path,
                  smiles: str = "") -> bool:
    """
    Convert an in-memory SDF mol block to PDBQT using the Open Babel Python API.

    Steps:
      1. pybel.readstring("sdf", ...) — parse the RDKit-generated SDF block
      2. OBChargeModel("gasteiger").ComputeCharges() — assign partial charges
      3. mol.write("pdbqt", ...) — write the PDBQT file

    No subprocess, no temp files. Returns True if the file was written.
    """
    try:
        # Append the SDF record terminator so OB's reader gets a clean entry
        mol = pybel_ob.readstring("sdf", sdf_block + "\n$$$$\n")
        mol.OBMol.SetTitle(zinc_id)

        # Gasteiger partial charges — equivalent to --partialcharge gasteiger
        charge_model = ob.OBChargeModel.FindType("gasteiger")
        if charge_model is not None:
            charge_model.ComputeCharges(mol.OBMol)

        mol.write("pdbqt", str(pdbqt_path), overwrite=True)

        # Record identity in the file header (Vina ignores unknown REMARKs, and
        # docking_engine copies them into the output pose file).
        if smiles:
            body = pdbqt_path.read_text()
            pdbqt_path.write_text(
                f"REMARK ZINC_ID {zinc_id}\nREMARK SMILES {smiles}\n" + body)

        return pdbqt_path.exists() and pdbqt_path.stat().st_size > 0
    except Exception as exc:
        log.debug("OB conversion failed for %s: %s", zinc_id, exc)
        return False


def embed_molecule(zinc_id: str, smi: str, subdir: str) -> dict:
    """
    Embed one SMILES into 3D via RDKit, then write PDBQT via Open Babel
    Python API. No subprocess calls, no temp SDF files.
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

    # Add explicit Hs so 3D embedding places them correctly; set ZINC ID as name
    mol = Chem.AddHs(mol)
    mol.SetProp("_Name", zinc_id)
    mol.SetProp("_SMILES", smi)

    params            = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1

    try:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)
    except Exception as exc:
        log.debug("EmbedMultipleConfs failed for %s: %s", zinc_id, exc)
        result.update(status="embed_failed")
        return result

    if not cids:
        result.update(status="embed_failed")
        return result

    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94s", maxIters=FF_STEPS)
        # Serialise the 3D conformer to an SDF block in memory — no temp file
        sdf_block = Chem.MolToMolBlock(mol, confId=int(cids[0]))
    except Exception as exc:
        log.debug("MMFF/serialisation failed for %s: %s", zinc_id, exc)
        result.update(status="ff_failed")
        return result

    # Convert SDF block → PDBQT via Open Babel Python API
    if _sdf_to_pdbqt(sdf_block, zinc_id, pdbqt_path, smiles=smi):
        result.update(path=str(pdbqt_path))
    else:
        result.update(status="ob_pdbqt_failed")

    return result


def process_batch(args: tuple) -> list:
    """Worker entry point: (batch of (zinc_id, smi), subdir_str)"""
    batch, subdir = args
    return [embed_molecule(zinc_id, smi, subdir) for zinc_id, smi in batch]


def process_triple_batch(triples: list) -> list:
    """Worker entry point: list of (zinc_id, smi, subdir) triples."""
    return [embed_molecule(z, s, d) for z, s, d in triples]


# ── Helpers ───────────────────────────────────────────────────────────────────

def subdir_for_index(base: Path, idx: int, subdir_size: int) -> Path:
    """Return e.g. base/0000000-0009999/ for idx 0..9999."""
    start = (idx // subdir_size) * subdir_size
    end   = start + subdir_size - 1
    return base / f"{start:07d}-{end:07d}"


def load_pairs(path: str, zinc_col: str, smiles_col: str,
               fmt: str = "auto") -> list:
    """
    Load (zinc_id, smiles) pairs from either:

      * CSV  — as produced by 01_extract_top_hits.py, with named columns
      * SMI  — whitespace-delimited, one molecule per line:
                   <SMILES> <zinc_id>
               e.g.  O=C(NCc1ccc(N2CCOCC2)c(F)c1)N1CCC(...)CC1 58571844

    fmt is "auto" (decide from the extension / first line), "csv" or "smi".
    Lines that are blank, commented (#) or missing a second field are skipped.
    """
    suffix = Path(path).suffix.lower()

    if fmt == "auto":
        if suffix in (".smi", ".smiles", ".ism", ".txt"):
            fmt = "smi"
        elif suffix in (".csv", ".tsv"):
            fmt = "csv"
        else:
            # Sniff the first non-empty line: a comma means CSV.
            with open(path) as fh:
                first = next((ln for ln in fh if ln.strip()), "")
            fmt = "csv" if "," in first else "smi"

    if fmt == "csv":
        df = pd.read_csv(path, usecols=[zinc_col, smiles_col])
        df = df.dropna().reset_index(drop=True)
        return list(zip(df[zinc_col].astype(str), df[smiles_col].astype(str)))

    # ── SMI: whitespace-delimited SMILES + ZINC ID ───────────────────────────
    pairs, skipped = [], 0
    seen_header    = False
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                skipped += 1
                log.debug("Line %d: fewer than 2 fields, skipped", lineno)
                continue
            smi, zinc_id = fields[0], fields[1]
            # Tolerate a header row such as "smiles zinc_id"
            if (not seen_header and lineno == 1
                    and Chem.MolFromSmiles(smi) is None):
                seen_header = True
                log.info("Treating first line as a header: %r", line)
                continue
            pairs.append((zinc_id, smi))

    if skipped:
        log.warning("Skipped %d malformed line(s) in %s", skipped, path)
    return pairs


def main() -> None:
    if not OB_AVAILABLE:
        raise SystemExit(
            "ERROR: Open Babel Python bindings not found.\n"
            "Install with:  conda install -c conda-forge openbabel\n"
            "Then verify:   python -c 'from openbabel import pybel; print(pybel.__version__)'"
        )

    parser = argparse.ArgumentParser(
        description="Convert top-hit SMILES to 3D PDBQT files for AutoDock Vina"
    )
    parser.add_argument("--input",        "-i", required=True,
                        help="CSV from 01_extract_top_hits.py, or a "
                             ".smi file of '<SMILES> <zinc_id>' lines")
    parser.add_argument("--format",       "-f", choices=("auto", "csv", "smi"),
                        default="auto",
                        help="Input format (default: auto-detect from the "
                             "extension, falling back to sniffing line 1)")
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

    pairs = load_pairs(args.input, args.zinc_col, args.smiles_col, args.format)
    if not pairs:
        raise SystemExit(f"ERROR: no usable compounds found in {args.input}")
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
