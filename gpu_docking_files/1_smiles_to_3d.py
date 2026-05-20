"""
Step 1: SMILES → 3D SDF Conversion
────────────────────────────────────
Reads a file of SMILES strings (with optional ZINC IDs), generates
3D conformers via RDKit + ETKDG, and writes individual .pdbqt files
(ready for AutoDock) plus a master SDF for inspection.

Expected input file format (tab or space delimited):
    ZINC000001234567  CC(=O)Oc1ccccc1C(=O)O
    ZINC000001234568  c1ccccc1
    ...
Alternatively, SMILES-only (one per line) — IDs auto-assigned.

Usage:
    python 1_smiles_to_3d.py \
        --input compounds.smi \
        --outdir ligands_3d \
        --workers 8
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_MW          = 800        # Daltons – skip huge molecules
MAX_ROTATABLE   = 15         # bonds – very flexible mols dock poorly
NUM_CONFORMERS  = 3          # how many 3-D conformers to embed
FF_STEPS        = 500        # MMFF optimisation steps


# ── Helpers ──────────────────────────────────────────────────────────────────
def standardise(mol: Chem.Mol) -> Chem.Mol | None:
    """Neutralise, pick largest fragment, canonicalise."""
    try:
        clean = rdMolStandardize.Cleanup(mol)
        largest = rdMolStandardize.LargestFragmentChooser().choose(clean)
        uncharge = rdMolStandardize.Uncharger().uncharge(largest)
        return uncharge
    except Exception:
        return None


def embed_molecule(zinc_id: str, smi: str, outdir: Path) -> dict:
    """
    Embed a single SMILES into 3-D, optimise, write PDBQT.
    Returns a result dict for logging / error tracking.
    """
    result = {"zinc_id": zinc_id, "smiles": smi, "status": "ok", "path": None}

    # --- Parse ---------------------------------------------------------------
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        result.update(status="invalid_smiles")
        return result

    # --- Filter --------------------------------------------------------------
    mw = Descriptors.ExactMolWt(mol)
    if mw > MAX_MW:
        result.update(status=f"too_heavy_{mw:.0f}Da")
        return result

    rot = AllChem.CalcNumRotatableBonds(mol)
    if rot > MAX_ROTATABLE:
        result.update(status=f"too_flexible_{rot}_bonds")
        return result

    # --- Standardise ---------------------------------------------------------
    mol = standardise(mol)
    if mol is None:
        result.update(status="standardise_failed")
        return result

    # --- Add Hs & embed ------------------------------------------------------
    mol = Chem.AddHs(mol)
    mol.SetProp("_Name", zinc_id)
    mol.SetProp("ZINC_ID", zinc_id)
    mol.SetProp("SMILES", smi)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1  # keep 1 per worker; outer pool handles parallelism

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFORMERS, params=params)
    if not cids:
        result.update(status="embed_failed")
        return result

    # --- MMFF minimisation ---------------------------------------------------
    AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94s", maxIters=FF_STEPS)

    # --- Write SDF -----------------------------------------------------------
    ## Why do we need SDF??
    sdf_path = outdir / f"{zinc_id}.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol, confId=int(cids[0]))
    writer.close()

    # --- Convert SDF → PDBQT via Open Babel (if available) ------------------
    pdbqt_path = outdir / f"{zinc_id}.pdbqt"
    obabel_cmd = (
        f"obabel {sdf_path} -O {pdbqt_path} --partialcharge gasteiger -h 2>/dev/null"
    )
    ## Uses Obabel - Can we not use rdkit for this? -> Can it generate partial gasteiger charges?
    ## Meeko alternative? -> If you use Obabel - this should be fine - there shouldn't be (though you can check for )
    ## Zero - coordinates, since you are using rdkit to generate the molecules...
    ## Take in certain compounds, generate the 3D conformer, then output_old the file as sdf.
    ## The sdf file is converted using obabel.
    ## This process could be parallelised...

    ret = os.system(obabel_cmd)
    if ret != 0:
        # Fallback: write a plain PDB for manual conversion
        pdb_path = outdir / f"{zinc_id}.pdb"
        Chem.MolToPDBFile(mol, str(pdb_path), confId=int(cids[0]))
        result.update(
            status="ok_no_pdbqt",
            path=str(pdb_path),
            note="obabel not found; wrote .pdb instead",
        )
    else:
        result.update(path=str(pdbqt_path))

    return result


def parse_smiles_file(filepath: str) -> list[tuple[str, str]]:
    """
    Parse the input file.  Accepts:
      - ZINC_ID  SMILES
      - SMILES   ZINC_ID   (guessed by checking which field is valid SMILES)
      - SMILES only        (auto-assigns ZINC_ID as 'mol_NNNN')
    """
    pairs = []
    with open(filepath) as fh:
        for i, raw in enumerate(fh):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                pairs.append((f"mol_{i:07d}", parts[0]))
            elif len(parts) >= 2:
                # Heuristic: the longer token is usually the SMILES
                a, b = parts[0], parts[1]
                if Chem.MolFromSmiles(a) is not None and Chem.MolFromSmiles(b) is None:
                    pairs.append((b, a))   # (ID, SMILES)
                else:
                    pairs.append((a, b))   # assume first=ID second=SMILES
    return pairs


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SMILES → 3D PDBQT converter")
    parser.add_argument("--input",   required=True, help="Input .smi file")
    parser.add_argument("--outdir",  default="ligands_3d", help="Output directory")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Parallel worker processes")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = parse_smiles_file(args.input)
    log.info(f"Loaded {len(pairs)} compounds from {args.input}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(embed_molecule, zid, smi, outdir): zid
            for zid, smi in pairs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if i % 500 == 0 or i == len(pairs):
                ok = sum(1 for r in results if "ok" in r["status"])
                log.info(f"  {i}/{len(pairs)}  ✓ {ok} embedded")

    # Write summary
    ok      = [r for r in results if "ok" in r["status"]]
    failed  = [r for r in results if "ok" not in r["status"]]
    log.info(f"\nDone.  Embedded: {len(ok)}  Failed: {len(failed)}")
    if failed:
        log.warning("Failed molecules:")
        for r in failed[:20]:
            log.warning(f"  {r['zinc_id']}  {r['status']}")

    # Write manifest for next step
    manifest = outdir / "manifest.txt"
    with open(manifest, "w") as fh:
        for r in ok:
            fh.write(f"{r['zinc_id']}\t{r['smiles']}\t{r['path']}\n")
    log.info(f"Manifest written to {manifest}")


if __name__ == "__main__":
    main()
