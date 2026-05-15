"""
Docking Score Predictor — Python 3.9 compatible, parallel fingerprint generation
=================================================================================
Pipeline:
  1. Convert SMILES -> Morgan Fingerprints (2048-bit, radius 2) via RDKit
     - Uses mpi4py if available and launched with mpirun (multiple ranks)
     - Falls back to multiprocessing.Pool (multiple CPU cores, single node)
     - Falls back to single-threaded if neither is usable
  2. Wrap data in a PyTorch Dataset / DataLoader
  3. Train a fully-connected neural network to predict docking scores
  4. Run inference on compounds with unknown docking scores

MPI usage:
    mpirun -n 8 python docking_score_predictor.py
    (rank 0 coordinates and runs training; all ranks share fingerprint work)

Single-node usage (multiprocessing):
    python docking_score_predictor.py
"""

import os
import math
import multiprocessing
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
import joblib


# ──────────────────────────────────────────────
# MPI initialisation (optional)
# ──────────────────────────────────────────────

try:
    from mpi4py import MPI as _MPI
    _MPI_COMM  = _MPI.COMM_WORLD
    _MPI_RANK  = _MPI_COMM.Get_rank()
    _MPI_SIZE  = _MPI_COMM.Get_size()
    # Only treat as truly distributed if more than one rank was launched
    MPI_AVAILABLE = _MPI_SIZE > 1
except ImportError:
    _MPI_COMM  = None
    _MPI_RANK  = 0
    _MPI_SIZE  = 1
    MPI_AVAILABLE = False

# MPI broadcast tags used to coordinate worker ranks
_TAG_FINGERPRINTS = 1
_TAG_STOP         = 0

# ──────────────────────────────────────────────
# Module-level worker (must be at top level for
# multiprocessing.Pool pickling to work)
# ──────────────────────────────────────────────

def _fp_worker(args: Tuple) -> Tuple[List[np.ndarray], List[int]]:
    """
    Compute Morgan fingerprints for a chunk of (idx, smiles) pairs.
    Designed to be called by multiprocessing.Pool.map or directly by MPI workers.

    Args:
        args: (chunk, radius, n_bits)
              chunk is a list of (original_index, smiles_string) tuples

    Returns:
        (fps, valid_indices) where fps is a list of float32 arrays
    """
    chunk, radius, n_bits = args
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps: List[np.ndarray] = []
    valid_idx: List[int]  = []

    for idx, smi in chunk:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = generator.GetFingerprintAsNumPy(mol).astype(np.float32)
        fps.append(fp)
        valid_idx.append(idx)

    return fps, valid_idx


def _split_into_chunks(items: List, n_chunks: int) -> List[List]:
    """Split a list into n_chunks roughly equal sublists."""
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i: i + chunk_size] for i in range(0, len(items), chunk_size)]


# ──────────────────────────────────────────────
# 1. Morgan Fingerprint Generation
# ──────────────────────────────────────────────

def generate_fingerprints(df: pd.DataFrame,
                           smiles_col: str = "SMILES",
                           radius: int = 2,
                           n_bits: int = 2048,
                           n_workers: Optional[int] = None) -> Tuple[np.ndarray, pd.Index]:
    """
    Generate Morgan fingerprints for every row in df, using:
      - MPI scatter/gather   if mpi4py is available AND >1 rank was launched
      - multiprocessing.Pool if running on a single node
      - single-threaded loop as a final fallback

    In MPI mode this function is COLLECTIVE — all ranks must call it together.
    Rank 0 returns (fingerprints, valid_idx); other ranks return (None, None).

    Args:
        df        : DataFrame containing SMILES strings
        smiles_col: column name for SMILES
        radius    : Morgan fingerprint radius
        n_bits    : fingerprint size in bits
        n_workers : number of processes for multiprocessing fallback
                    (defaults to os.cpu_count())

    Returns:
        fingerprints : np.ndarray of shape (N, n_bits)   [rank 0 only in MPI mode]
        valid_idx    : pd.Index of original df indices    [rank 0 only in MPI mode]
    """
    # Build the full list of (original_index, smiles) pairs on rank 0
    pairs: List[Tuple[int, str]] = list(zip(df.index.tolist(), df[smiles_col].tolist()))

    # ── MPI path ────────────────────────────────────────────────────────────
    if MPI_AVAILABLE:
        params = {"radius": radius, "n_bits": n_bits}

        if _MPI_RANK == 0:
            # Broadcast task signal and params so workers know what to do
            _MPI_COMM.bcast(_TAG_FINGERPRINTS, root=0)
            _MPI_COMM.bcast(params, root=0)

            # Split pairs into one chunk per rank and scatter
            chunks = _split_into_chunks(pairs, _MPI_SIZE)
            # Pad to exactly _MPI_SIZE chunks in case len(pairs) < _MPI_SIZE
            while len(chunks) < _MPI_SIZE:
                chunks.append([])
        else:
            chunks = None

        # Every rank receives its own chunk
        local_chunk = _MPI_COMM.scatter(chunks, root=0)
        local_fps, local_idx = _fp_worker((local_chunk, radius, n_bits))

        # Gather results back to rank 0
        all_fps = _MPI_COMM.gather(local_fps, root=0)
        all_idx = _MPI_COMM.gather(local_idx, root=0)

        if _MPI_RANK == 0:
            flat_fps = [fp for rank_fps in all_fps for fp in rank_fps]
            flat_idx = [i  for rank_idx in all_idx for i  in rank_idx]
            _report_skipped(pairs, flat_idx)
            return np.vstack(flat_fps), pd.Index(flat_idx)
        else:
            return None, None   # type: ignore[return-value]

    # ── Multiprocessing path ─────────────────────────────────────────────────
    #n_proc = n_workers or os.cpu_count() or 1
    # Don't spawn more workers than there are compounds
    #n_proc = min(n_proc, len(pairs))
    n_proc = 5

    if n_proc > 1:
        print("  [parallel] Using {} worker processes (multiprocessing)".format(n_proc))
        chunks    = _split_into_chunks(pairs, n_proc)
        pool_args = [(chunk, radius, n_bits) for chunk in chunks]

        with multiprocessing.Pool(processes=n_proc) as pool:
            results = pool.map(_fp_worker, pool_args)

        flat_fps = [fp for chunk_fps, _ in results for fp in chunk_fps]
        flat_idx = [i  for _, chunk_idx in results for i  in chunk_idx]

    else:
        # ── Single-threaded fallback ─────────────────────────────────────────
        print("  [serial] Single-threaded fingerprint generation")
        flat_fps, flat_idx = _fp_worker((pairs, radius, n_bits))

    _report_skipped(pairs, flat_idx)
    return np.vstack(flat_fps), pd.Index(flat_idx)


def _report_skipped(pairs: List[Tuple[int, str]], valid_idx: List[int]) -> None:
    """Print a warning for any SMILES that failed to parse."""
    valid_set = set(valid_idx)
    for idx, smi in pairs:
        if idx not in valid_set:
            print("  [!] Invalid SMILES at index {}: {!r} — skipped.".format(idx, smi))


def _mpi_worker_loop() -> None:
    """
    Non-rank-0 MPI ranks enter this loop after startup.
    They participate in collective operations (fingerprint scatter/gather)
    until rank 0 broadcasts _TAG_STOP.
    """
    while True:
        task   = _MPI_COMM.bcast(None, root=0)
        if task == _TAG_STOP:
            break
        params = _MPI_COMM.bcast(None, root=0)

        # Receive our chunk and compute
        local_chunk          = _MPI_COMM.scatter(None, root=0)
        local_fps, local_idx = _fp_worker(
            (local_chunk, params["radius"], params["n_bits"])
        )

        # Send results back
        _MPI_COMM.gather(local_fps, root=0)
        _MPI_COMM.gather(local_idx, root=0)


# ──────────────────────────────────────────────
# 2. PyTorch Datasets
# ──────────────────────────────────────────────

class MoleculeDataset(Dataset):
    """
    Dataset for labelled compounds (known docking scores).

    Args:
        fingerprints : np.ndarray (N, n_bits)
        scores       : np.ndarray (N,)  — docking scores
    """
    def __init__(self, fingerprints: np.ndarray, scores: np.ndarray):
        self.X = torch.tensor(fingerprints, dtype=torch.float32)
        self.y = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class UnlabelledMoleculeDataset(Dataset):
    """
    Dataset for inference — compounds with unknown docking scores.

    Args:
        fingerprints : np.ndarray (N, n_bits)
        zinc_ids     : list of ZINC IDs (optional, for bookkeeping)
    """
    def __init__(self,
                 fingerprints: np.ndarray,
                 zinc_ids: Optional[List[str]] = None):
        self.X = torch.tensor(fingerprints, dtype=torch.float32)
        self.zinc_ids = zinc_ids or [str(i) for i in range(len(fingerprints))]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        return self.X[idx], self.zinc_ids[idx]


# ──────────────────────────────────────────────
# 3. Neural Network
# ──────────────────────────────────────────────

class DockingScoreNet(nn.Module):
    """
    Fully-connected network:  2048 -> 1024 -> 512 -> 256 -> 1
    BatchNorm + Dropout after every hidden layer for regularisation.
    """
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
        layers.append(nn.Linear(prev, 1))   # regression head
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
# 4. Training Loop
# ──────────────────────────────────────────────

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                n_epochs: int = 50,
                lr: float = 1e-3,
                device: str = "cpu",
                save_path: str = "model_1_20April2026.pt") -> nn.Module:
    """
    Train model with MSE loss and Adam optimiser.
    Saves the best checkpoint (lowest validation loss).
    """
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()
    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print("Epoch {:>3}/{}  train MSE={:.4f}  val MSE={:.4f}".format(
            epoch, n_epochs, train_loss, val_loss))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print("             + Saved best model (val MSE={:.4f})".format(best_val))

    # Reload best weights
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# ──────────────────────────────────────────────
# 5. Inference
# ──────────────────────────────────────────────

def predict(model: nn.Module,
            df_unknown: pd.DataFrame,
            scaler: StandardScaler,
            smiles_col: str = "SMILES",
            zinc_col: str = "zinc_id",
            device: str = "cpu",
            batch_size: int = 256,
            n_workers: Optional[int] = None) -> pd.DataFrame:
    """
    Predict docking scores for compounds with unknown scores.
    Returns a DataFrame sorted by predicted score (best = most negative first).
    """
    fps, valid_idx = generate_fingerprints(
        df_unknown, smiles_col=smiles_col, n_workers=n_workers
    )
    fps_scaled = scaler.transform(fps)

    zinc_ids: Optional[List[str]] = (
        df_unknown.loc[valid_idx, zinc_col].tolist()
        if zinc_col in df_unknown else None
    )
    dataset = UnlabelledMoleculeDataset(fps_scaled, zinc_ids)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds: List[float] = []
    ids:   List[str]   = []

    with torch.no_grad():
        for X_batch, batch_ids in loader:
            out = model(X_batch.to(device)).squeeze(1).cpu().numpy()
            preds.extend(out.tolist())
            ids.extend(batch_ids)

    result = pd.DataFrame({
        zinc_col:                  ids,
        smiles_col:                df_unknown.loc[valid_idx, smiles_col].values,
        "predicted_docking_score": preds,
    })
    return result.sort_values("predicted_docking_score").reset_index(drop=True)


# ──────────────────────────────────────────────
# 6. Main Pipeline
# ──────────────────────────────────────────────

def run_pipeline(df_labelled: pd.DataFrame,
                 df_unknown: Optional[pd.DataFrame] = None,
                 smiles_col: str = "SMILES",
                 zinc_col: str = "zinc_id",
                 score_col: str = "LEScore",
                 n_bits: int = 2048,
                 radius: int = 2,
                 val_split: float = 0.1,
                 test_split: float = 0.1,
                 batch_size: int = 64,
                 n_epochs: int = 50,
                 lr: float = 1e-3,
                 dropout: float = 0.3,
                 hidden_dims: Tuple[int, ...] = (1024, 512, 256),
                 model_path: str = "model_1_20April2026.pt",
                 scaler_path: str = "scaler_01_20April2026.pkl",
                 n_workers: Optional[int] = None,
                 seed: int = 42) -> Dict[str, object]:
    """
    Full pipeline (rank 0 only in MPI mode):
        labelled data -> fingerprints -> scale -> train/val/test split
                      -> train model -> evaluate -> (optionally) predict unknowns

    Returns a dict with keys: model, scaler, test_rmse, predictions (if df_unknown given)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Parallelism report ────────────────────
    if MPI_AVAILABLE:
        print("Parallelism: MPI ({} ranks)".format(_MPI_SIZE))
    else:
        n_proc = n_workers or os.cpu_count() or 1
        print("Parallelism: multiprocessing ({} workers)".format(n_proc))

    # ── Device selection ──────────────────────
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print("  GPU {}: {}".format(i, torch.cuda.get_device_name(i)))
        device = torch.device("cuda:0")
        print("Using: {}".format(torch.cuda.get_device_name(device)))
    else:
        device = torch.device("cpu")
        print("No CUDA GPU found, using CPU")

    # ── 6a. Fingerprints ──────────────────────
    print("\nGenerating Morgan fingerprints for labelled data ...")
    fps, valid_idx = generate_fingerprints(
        df_labelled, smiles_col=smiles_col,
        radius=radius, n_bits=n_bits, n_workers=n_workers
    )
    scores = df_labelled.loc[valid_idx, score_col].values.astype(np.float32)
    print("  {} valid compounds out of {}\n".format(len(fps), len(df_labelled)))

    # ── 6b. Scale fingerprints ────────────────
    scaler = StandardScaler()
    fps_scaled = scaler.fit_transform(fps)
    joblib.dump(scaler, scaler_path)
    print("Scaler saved to '{}'".format(scaler_path))

    # ── 6c. Dataset splits ────────────────────
    full_ds = MoleculeDataset(fps_scaled, scores)
    n_total = len(full_ds)
    n_val   = int(n_total * val_split)
    n_test  = int(n_total * test_split)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    print("Split -> train={}, val={}, test={}\n".format(n_train, n_val, n_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # ── 6d. Model ─────────────────────────────
    model = DockingScoreNet(input_dim=n_bits, hidden_dims=hidden_dims, dropout=dropout)
    print("Model architecture:\n{}\n".format(model))

    # ── 6e. Train ─────────────────────────────
    model = train_model(
        model, train_loader, val_loader,
        n_epochs=n_epochs, lr=lr, device=device, save_path=model_path
    )

    # ── 6f. Test evaluation ───────────────────
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    all_preds: List[float] = []
    all_true:  List[float] = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            test_loss += criterion(out, y_batch).item() * len(X_batch)
            all_preds.extend(out.squeeze(1).cpu().numpy().tolist())
            all_true.extend(y_batch.squeeze(1).cpu().numpy().tolist())

    test_mse  = test_loss / len(test_loader.dataset)
    test_rmse = float(np.sqrt(test_mse))
    print("\nTest MSE:  {:.4f}".format(test_mse))
    print("Test RMSE: {:.4f}".format(test_rmse))

    result: Dict[str, object] = {
        "model":     model,
        "scaler":    scaler,
        "test_rmse": test_rmse,
    }

    # ── 6g. Predict unknowns ──────────────────
    if df_unknown is not None:
        print("\nGenerating fingerprints for unknown compounds ...")
        predictions = predict(
            model, df_unknown, scaler,
            smiles_col=smiles_col, zinc_col=zinc_col,
            device=device, n_workers=n_workers
        )
        predictions.to_csv("predictions.csv", index=False)
        print("Predictions saved to 'predictions.csv'")
        print(predictions.head(10).to_string(index=False))
        result["predictions"] = predictions

    return result


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    df_known = pd.read_csv("LE_Scores.csv")
    df_known = df_known.iloc[:200000, :]

    df_unknown = pd.read_csv("chembl_36_LE_Scores.csv")
    df_unknown = df_unknown.rename(columns={
        "chembl_id":        "zinc_id",
        "canonical_smiles": "SMILES",
    })
    df_unknown = df_unknown.iloc[:200000, :]

    results = run_pipeline(
        df_labelled = df_known,
        df_unknown  = df_unknown,
        smiles_col  = "SMILES",
        zinc_col    = "zinc_id",
        score_col   = "LEScore",
        n_epochs    = 20,
        batch_size  = 32,
        hidden_dims = (1024, 512, 256),
        dropout     = 0.3,
        val_split   = 0.1,
        test_split  = 0.1,
        n_workers = 5   # override CPU count for multiprocessing if desired
    )

    print("\nFinal test RMSE: {:.4f}".format(results["test_rmse"]))

    # Signal MPI workers to shut down (no-op if MPI not active)
    if MPI_AVAILABLE:
        _MPI_COMM.bcast(_TAG_STOP, root=0)


if __name__ == "__main__":
    # Guard needed for multiprocessing.Pool on Windows / macOS
    multiprocessing.freeze_support()

    if MPI_AVAILABLE and _MPI_RANK != 0:
        # Non-rank-0 MPI workers sit in this loop, participating in
        # fingerprint scatter/gather operations until rank 0 signals stop
        _mpi_worker_loop()
    else:
        main()
