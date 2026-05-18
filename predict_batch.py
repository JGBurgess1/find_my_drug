"""
Docking Score Predictor -- Inference Only
==========================================
Built for: 1x V100 GPU + 9 CPUs on CHPC

Pipeline per chunk:
  1. Load chunk of rows from .smi file
  2. Drop nulls and Boron-containing SMILES
  3. Generate Morgan fingerprints  (multiprocessing, 9 CPUs)
  4. Scale + run inference          (V100)
  5. Append results to output_old CSV
  6. Free memory, move to next chunk
"""

import os
import glob
import math
import multiprocessing
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
import joblib


# ------------------------------------------------------------------------------
# Configuration -- edit these before running
# ------------------------------------------------------------------------------

INPUT_DIR   = "../qcrb_docking_data/zinc_big_file"
OUTPUT_PATH = "output_old/predictions.csv"
MODEL_PATH  = "trained_models/model_1_20April2026.pt"
SCALER_PATH = "trained_models/scaler_01_20April2026.pkl"
FILE_GLOB   = "*.smi"

ZINC_COL    = "zinc_id"
SMILES_COL  = "smiles"

CHUNK_SIZE  = 50000
BATCH_SIZE  = 4096
N_WORKERS   = 6
RADIUS      = 2
N_BITS      = 2048


# ------------------------------------------------------------------------------
# Neural network -- must exactly match the saved architecture
# ------------------------------------------------------------------------------

class DockingScoreNet(nn.Module):
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


# ------------------------------------------------------------------------------
# Model + scaler loading
# ------------------------------------------------------------------------------

def load_model(model_path: str,
               scaler_path: str,
               input_dim: int = 2048,
               hidden_dims: Tuple[int, ...] = (1024, 512, 256),
               dropout: float = 0.3,
               device: torch.device = torch.device("cpu")
               ) -> Tuple[nn.Module, StandardScaler]:

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


# ------------------------------------------------------------------------------
# File discovery
# ------------------------------------------------------------------------------

def discover_files(input_dir: str, file_glob: str = "*.smi") -> List[str]:
    """
    Scan one level of child subdirectories and return sorted file paths.
    Files sitting directly in input_dir are ignored.
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


# ------------------------------------------------------------------------------
# Fingerprint generation
# ------------------------------------------------------------------------------

def _fp_worker(args: Tuple) -> Tuple[List[np.ndarray], List[str]]:
    """
    Module-level worker for multiprocessing -- must be picklable.
    Computes Morgan fingerprints for a list of (zinc_id, smiles) pairs.
    Returns only rows that parsed successfully.
    """
    chunk, radius, n_bits = args
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits)

    fps:      List[np.ndarray] = []
    zinc_ids: List[str]        = []

    for zinc_id, smi in chunk:
        mol = MolFromSmiles(smi)
        if mol is None:
            continue
        fp = generator.GetFingerprintAsNumPy(mol).astype(np.float32)
        fps.append(fp)
        zinc_ids.append(zinc_id)

    return fps, zinc_ids


def _split_into_chunks(items: List, n_chunks: int) -> List[List]:
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i: i + chunk_size] for i in range(0, len(items), chunk_size)]


def generate_fingerprints(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Generate Morgan fingerprints for a DataFrame using N_WORKERS CPU cores.
    Returns (fingerprint array, list of zinc_ids) for valid rows only.
    """
    pairs = list(zip(df[ZINC_COL].tolist(), df[SMILES_COL].tolist()))

    if N_WORKERS > 1:
        chunks    = _split_into_chunks(pairs, N_WORKERS)
        pool_args = [(chunk, RADIUS, N_BITS) for chunk in chunks]
        with multiprocessing.Pool(processes=N_WORKERS) as pool:
            results = pool.map(_fp_worker, pool_args)
        flat_fps = [fp  for chunk_fps,  _ in results for fp  in chunk_fps]
        flat_ids = [zid for _, chunk_ids in results for zid in chunk_ids]
    else:
        flat_fps, flat_ids = _fp_worker((pairs, RADIUS, N_BITS))

    n_invalid = len(pairs) - len(flat_fps)
    if n_invalid:
        print("    {:,} SMILES failed to parse and were skipped".format(n_invalid))

    return np.vstack(flat_fps), flat_ids


# ------------------------------------------------------------------------------
# PyTorch Dataset + GPU inference
# ------------------------------------------------------------------------------

class FingerprintDataset(Dataset):
    def __init__(self, fingerprints: np.ndarray):
        self.X = torch.tensor(fingerprints, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


def run_inference(model: nn.Module,
                  fps_scaled: np.ndarray,
                  device: torch.device) -> np.ndarray:
    """
    Pass scaled fingerprints through the model in GPU batches.
    Returns a (N,) float32 array of predicted docking scores.
    """
    loader = DataLoader(
        FingerprintDataset(fps_scaled),
        batch_size = BATCH_SIZE,
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


# ------------------------------------------------------------------------------
# Per-file chunked processing
# ------------------------------------------------------------------------------

def process_file_in_chunks(fpath:          str,
                            model:          nn.Module,
                            scaler:         StandardScaler,
                            device:         torch.device,
                            header_written: bool,
                            total_scored:   int
                            ) -> Tuple[bool, int, bool]:
    """
    Stream one file in chunks of CHUNK_SIZE rows.
    Peak memory = one chunk of fingerprints at a time (~400MB at 50k rows).

    Returns:
        header_written : updated flag (so next file knows not to re-write header)
        total_scored   : updated running total
        skipped        : True if the file could not be opened at all
    """
    fname = os.path.basename(fpath)

    try:
        reader = pd.read_table(
            fpath,
            sep       = r'\s+',
            engine    = 'python',
            chunksize = CHUNK_SIZE,
        )
    except Exception as exc:
        print("  [!] Could not open {}: {}".format(fname, exc))
        return header_written, total_scored, True

    chunk_idx   = 0
    file_scored = 0

    for chunk in reader:
        chunk_idx += 1

        # Normalise column names
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        if ZINC_COL not in chunk.columns or SMILES_COL not in chunk.columns:
            print("  [!] chunk {} -- expected columns not found, got: {}".format(
                chunk_idx, list(chunk.columns)))
            continue

        # Clean: drop nulls then drop Boron-containing SMILES
        before = len(chunk)
        chunk  = chunk.dropna(subset=[ZINC_COL, SMILES_COL])
        chunk = chunk[~chunk[SMILES_COL].str.contains(r'B(?!r)', regex=True, na=False)]
        chunk  = chunk.reset_index(drop=True)
        dropped = before - len(chunk)

        if len(chunk) == 0:
            print("  chunk {:>4}  |  all rows dropped".format(chunk_idx))
            continue
        if dropped:
            print("  chunk {:>4}  |  dropped {:,} rows (nulls / Boron)".format(
                chunk_idx, dropped))

        # Fingerprints across 9 CPUs
        fps, valid_ids = generate_fingerprints(chunk)

        if len(fps) == 0:
            print("  chunk {:>4}  |  no valid fingerprints".format(chunk_idx))
            del chunk
            continue

        # Scale + infer on V100
        fps_scaled = scaler.transform(fps)
        scores     = run_inference(model, fps_scaled, device)

        # Append results to output_old CSV
        result_slice = pd.DataFrame({
            ZINC_COL:                  valid_ids,
            SMILES_COL:                chunk.loc[
                                           chunk[ZINC_COL].isin(valid_ids),
                                           SMILES_COL].values,
            "predicted_docking_score": scores,
            "source_file":             fname,
        })

        result_slice.to_csv(
            OUTPUT_PATH,
            mode   = 'w' if not header_written else 'a',
            header = not header_written,
            index  = False,
        )
        header_written  = True
        file_scored    += len(result_slice)
        total_scored   += len(result_slice)

        print("  chunk {:>4}  |  {:>7,} scored  |  running total: {:,}".format(
            chunk_idx, file_scored, total_scored))

        del chunk, fps, fps_scaled, scores, result_slice

    print("  File total : {:,} compounds scored\n".format(file_scored))
    return header_written, total_scored, False


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.freeze_support()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("GPU : {}\n".format(torch.cuda.get_device_name(device)))
    else:
        device = torch.device("cpu")
        print("No GPU found -- using CPU\n")

    model, scaler = load_model(
        model_path  = MODEL_PATH,
        scaler_path = SCALER_PATH,
        device      = device,
    )

    all_files = discover_files(INPUT_DIR, file_glob=FILE_GLOB)
    n_files   = len(all_files)

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
        print("Removed previous output_old file '{}'\n".format(OUTPUT_PATH))

    header_written = False
    total_scored   = 0
    total_skipped  = 0

    for file_idx, fpath in enumerate(all_files, start=1):
        size_mb = os.path.getsize(fpath) / 1e6
        print("[{}/{}]  {}  ({:.1f} MB)".format(
            file_idx, n_files, os.path.basename(fpath), size_mb))

        header_written, total_scored, skipped = process_file_in_chunks(
            fpath          = fpath,
            model          = model,
            scaler         = scaler,
            device         = device,
            header_written = header_written,
            total_scored   = total_scored,
        )
        if skipped:
            total_skipped += 1

    print("=" * 55)
    print("Done.")
    print("  Files processed : {:,} / {:,}".format(
        n_files - total_skipped, n_files))
    print("  Total scored    : {:,}".format(total_scored))
    print("  Output          : {}".format(OUTPUT_PATH))
