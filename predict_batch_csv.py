"""
Docking Score Predictor -- Inference Only (CSV output, resumable)
=================================================================
Built for: 1x V100 GPU + 9 CPUs on CHPC

Features:
  - Saves predictions to a CSV file, appending as each chunk completes
  - JSON checkpoint file tracks the last processed file index for resuming
  - --start-idx / --end-idx flags allow manual splitting across GPU nodes

Single node (automatic resume):
    python predict_batch_csv.py

Split across two GPU nodes (run simultaneously on separate nodes):
    # Node 1
    python predict_batch_csv.py --start-idx 0    --end-idx 957  --output predictions_node1.csv
    # Node 2
    python predict_batch_csv.py --start-idx 958  --end-idx 1915 --output predictions_node2.csv

Other options:
    python predict_batch_csv.py --status     # print checkpoint position and exit
    python predict_batch_csv.py --restart    # clear checkpoint and output, start fresh
"""

import os
import sys
import glob
import json
import math
import argparse
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
# Configuration
# ------------------------------------------------------------------------------

CHECKPOINT_FILE  = "./output/predictions_checkpoint.json"
INPUT_DIR   = "/home/jburgess1/lustre/qcrb_docking_data/zinc20"
OUTPUT_PATH = "./output/predictions.csv"
MODEL_PATH  = "trained_models/model_18May_2026.pt"
SCALER_PATH = "trained_models/scaler_18May_2026.pkl"
FILE_GLOB   = "*.smi"

ZINC_COL         = "zinc_id"
SMILES_COL       = "smiles"

CHUNK_SIZE       = 100000
BATCH_SIZE       = 4096
N_WORKERS        = 9
RADIUS           = 2
N_BITS           = 2048


# ------------------------------------------------------------------------------
# Neural network
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
               device: torch.device) -> Tuple[nn.Module, StandardScaler]:
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found: {}".format(model_path))
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found: {}".format(scaler_path))

    model = DockingScoreNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    scaler = joblib.load(scaler_path)
    print("Model  loaded: '{}'".format(model_path))
    print("Scaler loaded: '{}'".format(scaler_path))
    return model, scaler


# ------------------------------------------------------------------------------
# File position checkpoint  (simple JSON file)
# ------------------------------------------------------------------------------

def save_checkpoint(checkpoint_file: str,
                    file_idx: int,
                    file_name: str,
                    total_scored: int) -> None:
    """
    Atomically write current position to the checkpoint JSON file.
    Called BEFORE processing each file so a crash mid-file restarts
    from that file on the next run.
    Uses a .tmp file + os.replace() to avoid a corrupt checkpoint
    if the process is killed during the write.
    """
    data = {
        "file_idx":     file_idx,
        "file_name":    file_name,
        "total_scored": total_scored,
    }
    tmp = checkpoint_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, checkpoint_file)


def load_checkpoint(checkpoint_file: str) -> Tuple[int, str, int]:
    """
    Read the checkpoint file.
    Returns (file_idx, file_name, total_scored).
    Returns (0, '', 0) if no checkpoint exists.
    """
    if not os.path.exists(checkpoint_file):
        return 0, "", 0
    try:
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
        return (
            int(data.get("file_idx",     0)),
            str(data.get("file_name",    "")),
            int(data.get("total_scored", 0)),
        )
    except Exception as exc:
        print("  [!] Could not read checkpoint: {} -- starting from 0".format(exc))
        return 0, "", 0


def clear_checkpoint(checkpoint_file: str) -> None:
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint '{}' cleared.".format(checkpoint_file))


def print_status(checkpoint_file: str) -> None:
    file_idx, file_name, total_scored = load_checkpoint(checkpoint_file)
    if file_name:
        print("Checkpoint: '{}'".format(checkpoint_file))
        print("  Last file index  : {:,}  (0-based)".format(file_idx))
        print("  Last file name   : {}".format(file_name))
        print("  Total scored     : {:,}".format(total_scored))
    else:
        print("No checkpoint found at '{}' -- not yet run.".format(
            checkpoint_file))


# ------------------------------------------------------------------------------
# File discovery
# ------------------------------------------------------------------------------

def discover_files(input_dir: str, file_glob: str = "*.smi") -> List[str]:
    """
    Sorted glob across one level of subdirectories.
    Both subdirs and files within each subdir are sorted, guaranteeing
    identical ordering on every run so the file_idx checkpoint is reliable.
    """
    subdirs = sorted([
        d for d in glob.glob(os.path.join(input_dir, "*"))
        if os.path.isdir(d)
    ])
    if not subdirs:
        raise FileNotFoundError(
            "No subdirectories found inside '{}'".format(input_dir))

    print("Subdirectories under '{}':".format(input_dir))
    all_files: List[str] = []
    for subdir in subdirs:
        matched = sorted(glob.glob(os.path.join(subdir, file_glob)))
        print("  {:<40s}  {:>4} file(s)".format(
            os.path.basename(subdir), len(matched)))
        all_files.extend(matched)

    if not all_files:
        raise FileNotFoundError(
            "No '{}' files found in subdirectories of '{}'".format(
                file_glob, input_dir))

    total_bytes = sum(os.path.getsize(f) for f in all_files)
    print("\nTotal files      : {:,}".format(len(all_files)))
    print("Total data size  : {:.2f} GB\n".format(total_bytes / 1e9))
    return all_files


# ------------------------------------------------------------------------------
# Fingerprint generation
# ------------------------------------------------------------------------------

def _fp_worker(args: Tuple) -> Tuple[List[np.ndarray], List[str]]:
    chunk, radius, n_bits = args
    gen  = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps: List[np.ndarray] = []
    ids: List[str]        = []
    for zinc_id, smi in chunk:
        mol = MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        fps.append(fp)
        ids.append(str(zinc_id))
    return fps, ids


def _split_into_chunks(items: List, n_chunks: int) -> List[List]:
    cs = math.ceil(len(items) / n_chunks)
    return [items[i: i + cs] for i in range(0, len(items), cs)]


def generate_fingerprints(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    pairs  = list(zip(df[ZINC_COL].tolist(), df[SMILES_COL].tolist()))
    n_proc = min(N_WORKERS, len(pairs))
    if n_proc > 1:
        chunks    = _split_into_chunks(pairs, n_proc)
        pool_args = [(c, RADIUS, N_BITS) for c in chunks]
        with multiprocessing.Pool(processes=n_proc) as pool:
            results = pool.map(_fp_worker, pool_args)
        flat_fps = [fp for fps, _  in results for fp in fps]
        flat_ids = [i  for _,  ids in results for i  in ids]
    else:
        flat_fps, flat_ids = _fp_worker((pairs, RADIUS, N_BITS))

    n_invalid = len(pairs) - len(flat_fps)
    if n_invalid:
        print("    {:,} invalid SMILES skipped".format(n_invalid))
    return np.vstack(flat_fps), flat_ids


# ------------------------------------------------------------------------------
# PyTorch inference
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
                            output_path:    str,
                            header_written: bool,
                            total_scored:   int
                            ) -> Tuple[bool, int, bool]:
    """
    Stream one file in chunks of CHUNK_SIZE rows.
    Returns (header_written, total_scored, skipped).
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
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        if ZINC_COL not in chunk.columns or SMILES_COL not in chunk.columns:
            print("  [!] chunk {} -- expected columns not found: {}".format(
                chunk_idx, list(chunk.columns)))
            continue

        before  = len(chunk)
        chunk   = chunk.dropna(subset=[ZINC_COL, SMILES_COL])
        chunk   = chunk[~chunk[SMILES_COL].str.contains(
            r'B(?!r)', regex=True, na=False)]
        chunk   = chunk.reset_index(drop=True)
        dropped = before - len(chunk)

        if len(chunk) == 0:
            continue
        if dropped:
            print("  chunk {:>4}  |  dropped {:,} rows".format(
                chunk_idx, dropped))

        fps, valid_ids = generate_fingerprints(chunk)
        if len(fps) == 0:
            del chunk
            continue

        fps_scaled = scaler.transform(fps)
        scores     = run_inference(model, fps_scaled, device)

        result = pd.DataFrame({
            ZINC_COL:                  valid_ids,
            SMILES_COL:                chunk.loc[
                                           chunk[ZINC_COL].isin(valid_ids),
                                           SMILES_COL].values,
            "predicted_docking_score": scores,
            "source_file":             fname,
        })

        result.to_csv(
            output_path,
            mode   = 'w' if not header_written else 'a',
            header = not header_written,
            index  = False,
        )
        header_written  = True
        file_scored    += len(result)
        total_scored   += len(result)

        print("  chunk {:>4}  |  {:>8,} scored  |  running total: {:,}".format(
            chunk_idx, file_scored, total_scored))

        del chunk, fps, fps_scaled, scores, result

    print("  File total : {:,} compounds scored\n".format(file_scored))
    return header_written, total_scored, False


# ------------------------------------------------------------------------------
# Post-run merge helper (combine outputs from multiple nodes)
# ------------------------------------------------------------------------------

def merge_node_outputs(output_paths: List[str],
                       merged_path: str) -> None:
    """
    Concatenate CSV files from multiple nodes into one sorted output.
    Call manually after all nodes have finished.

    Example:
        python predict_batch_csv.py --merge \\
            predictions_node1.csv predictions_node2.csv \\
            --output predictions_combined.csv
    """
    print("Merging {:,} files ...".format(len(output_paths)))
    frames = []
    for p in output_paths:
        df = pd.read_csv(p)
        frames.append(df)
        print("  {:>12,} rows  <--  {}".format(len(df), p))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        "predicted_docking_score").reset_index(drop=True)
    combined.to_csv(merged_path, index=False)

    print("\nMerged output   : {}".format(merged_path))
    print("Total rows      : {:,}".format(len(combined)))
    print("Best score      : {:.4f}".format(
        combined["predicted_docking_score"].iloc[0]))


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    multiprocessing.freeze_support()

    # Resolve output path and matching checkpoint file
    # Each node uses its own output + checkpoint so they never collide
    output_path     = args.output
    checkpoint_file = output_path.replace(".csv", "_checkpoint.json")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("GPU : {}\n".format(torch.cuda.get_device_name(device)))
    else:
        device = torch.device("cpu")
        print("No GPU found -- using CPU\n")

    # Handle --restart
    if args.restart:
        print("--restart: clearing checkpoint and output file ...")
        clear_checkpoint(checkpoint_file)
        if os.path.exists(output_path):
            os.remove(output_path)
            print("Removed '{}'".format(output_path))
        print()

    # Determine start index:
    #   --start-idx overrides the checkpoint (use for manual node splitting)
    #   otherwise fall back to whatever the checkpoint saved
    if args.start_idx is not None:
        start_idx    = args.start_idx
        prev_scored  = 0
        print("--start-idx override: starting from file index {:,}".format(
            start_idx))
    else:
        start_idx, last_file, prev_scored = load_checkpoint(checkpoint_file)
        if last_file:
            print("Resuming from checkpoint:")
            print("  File index   : {:,}".format(start_idx))
            print("  Last file    : {}".format(last_file))
            print("  Prev scored  : {:,}\n".format(prev_scored))

    # Determine end index (inclusive, 0-based)
    end_idx = args.end_idx   # None means process to the end of the list

    # Load model + scaler
    model, scaler = load_model(MODEL_PATH, SCALER_PATH, device)
    print()

    # Discover files -- sorted glob, same order every run
    all_files = discover_files(INPUT_DIR, file_glob=FILE_GLOB)
    n_files   = len(all_files)

    # Validate index bounds
    if start_idx >= n_files:
        print("start_idx={} >= total files={} -- nothing to do.".format(
            start_idx, n_files))
        return

    effective_end = end_idx if end_idx is not None else n_files - 1
    print("Processing files {:,} to {:,}  (of {:,} total)\n".format(
        start_idx, effective_end, n_files))

    # Header already written if resuming into an existing output file
    header_written = (
        start_idx > 0
        and args.start_idx is None          # only when using checkpoint resume
        and os.path.exists(output_path)
        and os.path.getsize(output_path) > 0
    )

    total_scored  = prev_scored
    total_skipped = 0

    for file_idx, fpath in enumerate(all_files, start=0):

        # Skip files before our start
        if file_idx < start_idx:
            continue

        # Stop after our end
        if end_idx is not None and file_idx > end_idx:
            print("Reached --end-idx {}  -- stopping.".format(end_idx))
            break

        fname   = os.path.basename(fpath)
        size_mb = os.path.getsize(fpath) / 1e6

        print("[{}/{}]  {}  ({:.1f} MB)".format(
            file_idx + 1, n_files, fname, size_mb))

        # Save position BEFORE processing -- crash mid-file restarts here
        save_checkpoint(checkpoint_file, file_idx, fname, total_scored)

        header_written, total_scored, skipped = process_file_in_chunks(
            fpath          = fpath,
            model          = model,
            scaler         = scaler,
            device         = device,
            output_path    = output_path,
            header_written = header_written,
            total_scored   = total_scored,
        )

        if skipped:
            total_skipped += 1

    # Mark completion in checkpoint
    save_checkpoint(checkpoint_file,
                    effective_end + 1, "COMPLETE", total_scored)

    print("=" * 55)
    print("Run complete.")
    print("  File range       : {:,} -- {:,}".format(start_idx, effective_end))
    print("  Files processed  : {:,}".format(
        effective_end - start_idx + 1 - total_skipped))
    print("  Files skipped    : {:,}".format(total_skipped))
    print("  Total scored     : {:,}".format(total_scored))
    print("  Output           : {}".format(output_path))
    print("  Checkpoint       : {}".format(checkpoint_file))
    if end_idx is not None:
        print()
        print("When all nodes are done, merge outputs with:")
        print("  python predict_batch_csv.py --merge "
              "predictions_node1.csv predictions_node2.csv "
              "--output predictions_combined.csv")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resumable docking score batch inference (CSV output)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single node, automatic resume
  python predict_batch_csv.py

  # Two nodes splitting 1916 files
  python predict_batch_csv.py --start-idx 0    --end-idx 957  --output predictions_node1.csv
  python predict_batch_csv.py --start-idx 958  --end-idx 1915 --output predictions_node2.csv

  # Merge after both nodes finish
  python predict_batch_csv.py --merge predictions_node1.csv predictions_node2.csv \\
      --output predictions_combined.csv

  # Check progress
  python predict_batch_csv.py --status

  # Start completely fresh
  python predict_batch_csv.py --restart
        """)

    parser.add_argument(
        "--start-idx", type=int, default=None,
        help="0-based file index to start from. "
             "Overrides checkpoint. Use to split work across nodes.")

    parser.add_argument(
        "--end-idx", type=int, default=None,
        help="0-based file index to stop at (inclusive). "
             "Use with --start-idx to partition work across nodes.")

    parser.add_argument(
        "--output", type=str, default=OUTPUT_PATH,
        help="Output CSV path (default: {})".format(OUTPUT_PATH))

    parser.add_argument(
        "--restart", action="store_true",
        help="Clear checkpoint and output file, start from scratch")

    parser.add_argument(
        "--status", action="store_true",
        help="Print checkpoint position and exit")

    parser.add_argument(
        "--merge", nargs="+", metavar="FILE",
        help="Merge these CSV files into --output and exit (no inference)")

    args = parser.parse_args()

    if args.status:
        checkpoint_file = args.output.replace(".csv", "_checkpoint.json")
        print_status(checkpoint_file)
        sys.exit(0)

    if args.merge:
        merge_node_outputs(args.merge, args.output)
        sys.exit(0)

    main(args)
