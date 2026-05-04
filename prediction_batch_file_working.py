import torch, os, glob, joblib, pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from rdkit.Chem import MolFromSmiles, AllChem, rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
import joblib
from multiprocessing import Pool, Process

class DockingScoreNet(nn.Module):
    """Default: 2048 -> 1024 -> 512 -> 256 -> 1"""
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dims: Tuple[int, ...] = (1024, 512, 256),
                 dropout: float = 0.3):
        super(DockingScoreNet, self).__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_model(model_path: str,
               scaler_path: str,
               input_dim: int = 2048,
               hidden_dims: Tuple[int, ...] = (1024, 512, 256),
               dropout: float = 0.3,
               device: torch.device = torch.device("cpu")
               ) -> Tuple[nn.Module, object]:
    """
    Load saved model weights and fitted scaler from disk.
    model.eval() is called immediately to disable dropout / batchnorm train mode.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found: {}".format(model_path))
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found: {}".format(scaler_path))

    model = DockingScoreNet(input_dim=input_dim,
                            hidden_dims=hidden_dims,
                            dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    scaler = joblib.load(scaler_path)
    print("Model  loaded from '{}'".format(model_path))
    print("Scaler loaded from '{}'".format(scaler_path))
    return model, scaler

def discover_files(input_dir: str, file_glob: str = "*.smi") -> List[str]:
    """
    Scan one level of child subdirectories inside input_dir and return a
    sorted list of all matching file paths. Files sitting directly in
    input_dir itself are ignored.

    Expected layout:
        input_dir/
            batch_001/
                part_aa.csv
                part_ab.csv
            batch_002/
                part_aa.csv
    """
    subdirs = sorted([
        d for d in glob.glob(os.path.join(input_dir, "*"))
        if os.path.isdir(d)
    ])

    if not subdirs:
        raise FileNotFoundError(
            "No subdirectories found inside '{}'".format(input_dir))

    print("Subdirectories found under '{}':".format(input_dir))
    all_files: List[str] = []
    for subdir in subdirs:
        matched = sorted(glob.glob(os.path.join(subdir, file_glob)))
        print("  {:<40s}  {:>4} file(s)".format(
            os.path.basename(subdir), len(matched)))
        all_files.extend(matched)

    if not all_files:
        raise FileNotFoundError(
            "No '{}' files found in any subdirectory of '{}'".format(
                file_glob, input_dir))

    total_bytes = sum(os.path.getsize(f) for f in all_files)
    print("\nFiles to process : {:,}".format(len(all_files)))
    print("Total data size  : {:.2f} GB\n".format(total_bytes / 1e9))
    return all_files

# Not necessary to include these 'aliases'

_ID_ALIASES = {
    "zinc_id",
}
_SMILES_ALIASES = {
    "smiles",
}

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    pass

# The meat of this program: Load a large file in batches.

def load_data_chunk(chunk_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    before = len(chunk_df)
    chunk_df = chunk_df.dropna().reset_index(drop=True)
    chunk_df = chunk_df[~chunk_df["smiles"].str.contains('B', na=False)]
    dropped = before - len(chunk_df)
    if dropped:
        print("  Dropped {:,} rows with missing values or B atoms".format(dropped))
    return chunk_df

def load_single_file(fpath: str,
                     zinc_col: str,
                     smiles_col: str) -> Optional[pd.DataFrame]:
    """
    Load one file, auto-detect its delimiter (comma or whitespace),
    normalise column names, drop rows with missing values, and return
    a two-column DataFrame.  Returns None on any unrecoverable error.
    """
    fname = os.path.basename(fpath)
    try:
        df = pd.read_table(fpath, sep=r'\s+', engine='python')
    except Exception as exc:
        print("  [!] Could not read {}: {}".format(fname, exc))
        return None

    before = len(df)
    df = df.dropna(subset=[zinc_col, smiles_col]).reset_index(drop=True)
    # Drop Boron rows?!
    # memory allocation?! # Is this done for me, in Python...
    df = df[~df[smiles_col].str.contains(r'B(?!r)', na=False)]
    # Remove the rows containing Boron rows.
    # Can add a log later.
    # B must not be in the SMILES column.
    dropped = before - len(df)
    if dropped:
        print("  Dropped {:,} rows with missing values or B atoms".format(dropped))
    return df

def generate_fingerprints(df: pd.DataFrame,
                           smiles_col: str = "SMILES",
                           radius: int = 2,
                           n_bits: int = 2048) -> Tuple[np.ndarray, pd.Index]:
    """
    Generate Morgan fingerprints for every row in df.

    Returns:
        fingerprints : np.ndarray of shape (N, n_bits)
        valid_idx    : index of rows where SMILES parsed successfully
    """
    # don't want this 'valid_idx' thing. - Keep the ZINC IDS?!
    # Try to generate fingerprints for the molecules.
    # If any fail, then ok ...
    fps: List[np.ndarray] = []
    indices: List[int] = []
    # ---- Uncomment below for including newer zinc IDs.
    # indices = List[str]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for row in df.itertuples():
        try:
            mol = MolFromSmiles(row[1])
            fp = generator.GetFingerprintAsNumPy(mol)
            fps.append(fp)
            indices.append(row[2])
        except Exception as e:
            print("  [!] Could not parse {}: {}".format(row[1], e))

    return np.vstack(fps), pd.Index(indices)

# Want to modify this to receive a different value.
class FingerprintDataset(Dataset):
    def __init__(self, fingerprints: np.ndarray):
        self.X = torch.tensor(fingerprints, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]

#Good, this works.

def run_inference(model: nn.Module,
                  fps_scaled: np.ndarray,
                  device: torch.device,
                  batch_size: int = 4096) -> np.ndarray:
    """
    Pass scaled fingerprints through the model in GPU batches.
    Returns a (N,) float32 array of predicted docking scores.
    """
    loader = DataLoader(
        FingerprintDataset(fps_scaled),
        batch_size = batch_size,
        shuffle    = False,
        num_workers= 0,
        pin_memory = (device.type == "cuda"),
    )
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch.to(device)).squeeze(1).cpu().numpy()
            preds.append(out)
    return np.concatenate(preds)

#### (1) Work on this part
#### Revise according to my needs,
#### See how to optimize, and how to proceed.
####

#### (2) Add MPI / multiprocessing, for the loading, and sending to the GPU.
#### Will cut down on time by almost 10 fold.
#### Significant speed -up...
def process_file_in_chunks(fpath:          str,
                            model:          nn.Module,
                            scaler:         StandardScaler,
                            device:         torch.device,
                            output_path:    str,
                            zinc_col:       str   = "zinc_id",
                            smiles_col:     str   = "smiles",
                            chunk_size:     int   = 50000,
                            batch_size:     int   = 4096,
                            header_written: bool  = False,
                            total_scored:   int   = 0
                            ) -> Tuple[bool, int, bool]:
    """
    Stream one file in chunks of chunk_size rows.
    For each chunk:
        1. Clean (drop nulls, drop Boron-containing SMILES)
        2. Generate Morgan fingerprints
        3. Scale + run inference
        4. Append results to output_path immediately
        5. Delete chunk data before loading the next one

    Returns:
        header_written : bool   -- passed through so caller knows if header exists
        total_scored   : int    -- running total across all files
        skipped        : bool   -- True if the file could not be opened at all
    """
    fname = os.path.basename(fpath)

    # Open as a chunk iterator -- nothing is loaded into memory yet
    try:
        reader = pd.read_table(
            fpath,
            sep       = r'\s+',
            engine    = 'python',
            chunksize = chunk_size,
        )
    except Exception as exc:
        print("  [!] Could not open {}: {}".format(fname, exc))
        return header_written, total_scored, True   # skipped

    chunk_idx   = 0
    file_scored = 0

    for chunk in reader:
        chunk_idx += 1

        # ── Clean ────────────────────────────────────────────────────────────
        before = len(chunk)
        chunk  = chunk.dropna(subset=[zinc_col, smiles_col])
        chunk  = chunk[~chunk[smiles_col].str.contains(r'B(?!r)', na=False)]
        chunk  = chunk.reset_index(drop=True)
        dropped = before - len(chunk)

        if len(chunk) == 0:
            print("  chunk {:>4}  |  all rows dropped".format(chunk_idx))
            continue

        if dropped:
            print("  chunk {:>4}  |  dropped {:,} rows (nulls / Boron)".format(
                chunk_idx, dropped))

        # ── Fingerprints ──────────────────────────────────────────────────────
        fps, valid_idx = generate_fingerprints(
            chunk,
            smiles_col = smiles_col,
            radius     = 2,
            n_bits     = 2048,
        )

        if len(fps) == 0:
            print("  chunk {:>4}  |  no valid fingerprints".format(chunk_idx))
            del chunk
            continue

        # ── Scale + infer ─────────────────────────────────────────────────────
        fps_scaled = scaler.transform(fps)
        scores     = run_inference(model, fps_scaled, device, batch_size=batch_size)

        # ── Write results for this chunk ──────────────────────────────────────
        result_slice = pd.DataFrame({
            zinc_col:                  valid_idx.tolist(),
            smiles_col:                chunk.loc[chunk[zinc_col].isin(valid_idx), smiles_col].values,
            "predicted_docking_score": scores,
            "source_file":             fname,
        })

        result_slice.to_csv(
            output_path,
            mode   = 'w' if not header_written else 'a',
            header = not header_written,
            index  = False,
        )
        header_written  = True
        file_scored    += len(result_slice)
        total_scored   += len(result_slice)

        print("  chunk {:>4}  |  {:>7,} scored  |  running total: {:,}".format(
            chunk_idx, file_scored, total_scored))

        # ── Free memory before next chunk ─────────────────────────────────────
        del chunk, fps, fps_scaled, scores, result_slice

    print("  File total: {:,} compounds scored\n".format(file_scored))
    return header_written, total_scored, False   # not skipped

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}\n".format(device))
    model, scaler = load_model(
        model_path="best_model.pt",
        scaler_path="scaler.pkl",
        device=device,
    )
    all_files = discover_files(
        input_dir="zinc_big_file/",
        file_glob="*.smi",
    )
    output_path     = "predictions.csv"
    header_written  = False
    total_scored    = 0
    total_skipped   = 0

    for file_idx, fpath in enumerate(all_files):
        size_mb = os.path.getsize(fpath) / 1e6
        print("[{}/{}]  {}  ({:.1f} MB)".format(
            file_idx, len(all_files), os.path.basename(fpath), size_mb))

        header_written, total_scored, skipped = process_file_in_chunks(
            fpath=fpath,
            model=model,
            scaler=scaler,
            device=device,
            output_path=output_path,
            chunk_size=50000,
            batch_size=4096,
            header_written=header_written,
            total_scored=total_scored,
        )
        if skipped:
            total_skipped += 1

    print("=" * 50)
    print("Done.  Total scored: {:,}".format(total_scored))
    if total_skipped:
        print("       Files skipped: {:,}".format(total_skipped))

if __name__ == "__main__":
    main()