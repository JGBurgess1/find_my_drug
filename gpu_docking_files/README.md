# Virtual Screening Pipeline — Setup & Usage
## SMILES → 3D Conformers → GPU Docking → CSV

---

## Files

| Script | Purpose |
|---|---|
| `1_smiles_to_3d.py` | Parse SMILES, embed 3D, write PDBQT + SDF (RDKit + Open Babel) |
| `2_gpu_docking.py` | GPU-parallel docking (UniDock / Vina-GPU / gnina / AutoDock-GPU) |
| `3_pipeline.py` | Ties both steps together; produces the final CSV |

---

## Dependencies

### Python
```bash
conda create -n vscreen python=3.11
conda activate vscreen
conda install -c conda-forge rdkit
pip install tqdm
```

### Open Babel (PDBQT conversion in step 1)
```bash
conda install -c conda-forge openbabel
```

---

## GPU Docking Backend — choose ONE

### Option A — Uni-Dock ⭐ (recommended — fastest, best for large libraries)

**Publication**: Yu et al., JCTC 2023. doi:10.1021/acs.jctc.2c01145
Achieves >1000× speedup vs. AutoDock Vina on a single CPU core.
Apache 2.0 licence (relicensed March 2025).

#### Install via conda (easiest):
```bash
conda install -c https://conda.mlops.dp.tech/caic -c conda-forge unidock
```

#### Build from source (CUDA ≥ 11.4, CMake ≥ 3.20):
```bash
git clone https://github.com/dptech-corp/Uni-Dock
cd Uni-Dock/unidock
cmake -B build
cmake --build build -j$(nproc)
cmake --install build        # adds `unidock` to the default PATH
```

#### Uni-Dock Tools (optional Python wrapper — enables SDF input):
```bash
pip install "git+https://github.com/dptech-corp/Uni-Dock.git#subdirectory=unidock_tools"
# Installs the `unidocktools` CLI command
```

#### How UniDock works in this pipeline (raw binary mode):

```
unidock \
  --receptor  protein.pdbqt \
  --gpu_batch ligand1.pdbqt ligand2.pdbqt ... \   ← PDBQT files, space-separated
  --center_x X --center_y Y --center_z Z \
  --size_x S  --size_y S  --size_z S \
  --search_mode balance \           ← fast / balance / detail
  --scoring vina \
  --num_modes 5 \
  --dir  output_dir/                ← writes <stem>_out.pdbqt per ligand
```

For batches > 500 ligands the pipeline automatically uses `--ligand_index`
(a text file listing PDBQT paths) instead of `--gpu_batch` to avoid shell
`ARG_MAX` limits.

Score is read from `REMARK VINA RESULT:` in each `*_out.pdbqt`.

#### Uni-Dock Tools mode (`--use_unidock_tools`):

```
unidocktools unidock_pipeline \
  -r protein.pdbqt \
  -i ligand_index.txt \    ← text file listing SDF paths (one per line)
  -sd output_dir/ \
  -cx X -cy Y -cz Z \
  -sx S -sy S -sz S \
  -sf vina -nm 5
```
This path handles ligand prep internally (no Open Babel needed).

#### GPU requirements:
- v1.1+: NVIDIA GPUs with compute capability ≥ 7.0 (V100, A100, RTX 3090+)
- v1.0: also supports AMD ROCm (no longer updated)

---

### Option B — Vina-GPU 2.1

```bash
git clone https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1
cd Vina-GPU-2.1
# Edit Makefile: set CUDA_PATH and OpenBabel paths
make
cp Vina-GPU-2-1 ~/.local/bin/
```

---

### Option C — AutoDock-GPU (fastest classical scoring, requires grid maps)

```bash
git clone https://github.com/ccsb-scripps/AutoDock-GPU
cd AutoDock-GPU
make DEVICE=CUDA NUMWI=128
cp bin/autodock_gpu_128wi ~/.local/bin/
# NOTE: requires pre-computed grid maps (.fld) from AutoGrid4
```

---

### Option D — gnina (CNN-based scoring, highest accuracy, slowest)

```bash
wget https://github.com/gnina/gnina/releases/latest/download/gnina \
     -O ~/.local/bin/gnina
chmod +x ~/.local/bin/gnina
```

---

## Receptor Preparation

The receptor must be a protonated, charge-assigned PDBQT file.

```bash
# Using ADFR Suite (recommended):
prepare_receptor -r protein.pdb -o protein.pdbqt -A hydrogens

# Using Open Babel:
obabel protein.pdb -O protein.pdbqt --partialcharge gasteiger -h
```

UniDock Tools can also accept PDB directly (`-r protein.pdb`).

---

## Docking Box Configuration (`box.conf`)

```ini
# Binding site centre (Å) — measure from PyMOL / UCSF Chimera
center_x = 12.5
center_y = -3.2
center_z = 44.1

# Box dimensions (Å)
size_x = 22
size_y = 22
size_z = 22

# Shared parameters
exhaustiveness = 8   # used by vina-gpu / gnina / autodock
num_modes = 5        # poses to report

# UniDock-specific (optional overrides)
scoring = vina       # vina / vinardo / ad4
```

---

## Input SMILES File Format

One compound per line, tab- or space-delimited:

```
ZINC000001234567   CC(=O)Oc1ccccc1C(=O)O
ZINC000001234568   c1ccccc1N
```

SMILES-only (no ID) also accepted — IDs auto-assigned as `mol_0000001`.

---

## Usage

### Full pipeline — UniDock (recommended)
```bash
python 3_pipeline.py \
    --input       compounds.smi \
    --receptor    protein.pdbqt \
    --config      box.conf \
    --backend     unidock \
    --gpus        0,1 \
    --batch_size  1200 \
    --search_mode balance \
    --workers     4 \
    --output_old      results.csv
```

### Full pipeline — UniDock Tools (SDF input, no manual ligand prep)
```bash
python 3_pipeline.py \
    --input             compounds.smi \
    --receptor          protein.pdbqt \
    --config            box.conf \
    --backend           unidock \
    --use_unidock_tools \
    --gpus              0 \
    --output_old            results.csv
```

### Full pipeline — Vina-GPU
```bash
python 3_pipeline.py \
    --input    compounds.smi \
    --receptor protein.pdbqt \
    --config   box.conf \
    --backend  vina-gpu \
    --gpus     0,1 \
    --workers  8 \
    --output_old   results.csv
```

### Step-by-step (for large libraries ≥ 100k — allows checkpointing)
```bash
# Step 1: embed 3D conformers
python 1_smiles_to_3d.py \
    --input   compounds.smi \
    --outdir  ligands_3d \
    --workers 32

# Step 2: dock (skip embed on resume with --skip_embed)
python 2_gpu_docking.py \
    --manifest   ligands_3d/manifest.txt \
    --receptor   protein.pdbqt \
    --config     box.conf \
    --backend    unidock \
    --gpus       0,1,2,3 \
    --batch_size 1200 \
    --search_mode balance \
    --workers    2 \
    --output_csv raw_results.csv
```

---

## Output CSV

```
zinc_id,smiles,docking_score
ZINC000001234567,CC(=O)Oc1ccccc1C(=O)O,-9.234
ZINC000001234568,c1ccccc1N,-7.801
```

Sorted ascending by `docking_score` (more negative = stronger predicted binding).

---

## Performance (approximate)

| Setup | Throughput |
|---|---|
| 1× A100 + UniDock (balance) | ~50,000–500,000 cpd/hr |
| 4× A100 + UniDock (balance) | ~200,000–2,000,000 cpd/hr |
| 1× A100 + UniDock (fast) | ~1,000,000+ cpd/hr |
| 1× A100 + AutoDock-GPU | ~50,000–200,000 cpd/hr |
| 1× A100 + Vina-GPU 2.1 | ~2,000–5,000 cpd/hr |
| 1× A100 + gnina | ~500–2,000 cpd/hr |

UniDock throughput scales with batch size — small batches have proportionally
more GPU setup overhead. Keep `--batch_size` at 1200 or higher on A100.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `unidock: command not found` | `conda install -c https://conda.mlops.dp.tech/caic -c conda-forge unidock` |
| `obabel not found` | `conda install -c conda-forge openbabel` |
| CUDA OOM | Reduce `--batch_size` (try 600 or 300) |
| Score returns `None` | Check `<batch_dir>/unidock.log`; ensure box is inside the binding site |
| `no_pdbqt_found` for many entries | Ensure Open Babel ran in step 1; check `ligands_3d/*.pdbqt` |
| `embed_failed` for many mols | Validate SMILES with `Chem.MolFromSmiles`; remove salts |
| All scores identical | Verify box centre coordinates and box size |
| ARG_MAX shell error | Shouldn't happen — pipeline auto-switches to `--ligand_index` at >500 |
