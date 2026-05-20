"""
Merge new Vina docking results with the original training set and retrain
the DockingScoreNet neural network for the next round of virtual screening.

Raw Vina scores are converted to Ligand Efficiency (LE) scores before merging:

    LE = (docking_score / n_heavy_atoms) * 100

This matches the LEScore column used in the original training data.

The Vina results CSV (written by Vina_Ona_Repo/results_writer.py) contains:
    ligand_name  score  success  [error  pose_file ...]
where ligand_name is the ZINC ID stem from the PDBQT filename.
SMILES are recovered from the top_1M_hits.csv produced by step 01.

Usage:
    python 03_merge_and_retrain.py \
        --original-data  LE_Scores.csv \
        --new-docking    docking_results_round2.csv \
        --smiles-map     top_1M_hits.csv \
        --output-model   trained_models/model_round2.pt \
        --output-scaler  trained_models/scaler_round2.pkl \
        --epochs         50
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split


# ── Model (matches docking_score_predictor_parallel.py architecture) ─────────

class DockingScoreNet(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=(1024, 512, 256), dropout=0.3):
        super().__init__()
        layers = []
        prev   = input_dim
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

    def forward(self, x):
        return self.net(x)


class MoleculeDataset(Dataset):
    def __init__(self, fps: np.ndarray, scores: np.ndarray):
        self.X = torch.tensor(fps,    dtype=torch.float32)
        self.y = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Ligand Efficiency conversion ─────────────────────────────────────────────

def docking_score_to_le(docking_score: float, smiles: str) -> float | None:
    """
    Convert a raw Vina docking score to a Ligand Efficiency score.

        LE = (docking_score / n_heavy_atoms) * 100

    Returns None if the SMILES cannot be parsed or has no heavy atoms.
    """
    mol = MolFromSmiles(str(smiles))
    if mol is None:
        return None
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy == 0:
        return None
    return (docking_score / n_heavy) * 100


# ── Fingerprint generation ────────────────────────────────────────────────────

def generate_fingerprints(smiles_list, radius=2, n_bits=2048) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (fps array, boolean valid_mask) so the caller can align scores.
    """
    generator  = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps        = []
    valid_mask = []

    for smi in smiles_list:
        mol = MolFromSmiles(str(smi))
        if mol is None:
            valid_mask.append(False)
            continue
        fps.append(generator.GetFingerprintAsNumPy(mol).astype(np.float32))
        valid_mask.append(True)

    return np.vstack(fps), np.array(valid_mask, dtype=bool)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_original_training_data(
    path: str, zinc_col: str, smiles_col: str, score_col: str
) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[zinc_col, smiles_col, score_col])
    df = df.rename(columns={
        zinc_col:   "zinc_id",
        smiles_col: "smiles",
        score_col:  "docking_score",
    })
    df = df.dropna().reset_index(drop=True)
    df["zinc_id"] = df["zinc_id"].astype(str)
    print(f"Original training data : {len(df):,} rows  ({path})")
    return df


def load_vina_results(results_path: str, smiles_map_path: str) -> pd.DataFrame:
    """
    Load Vina results CSV and join SMILES from the top-hits CSV.

    Auto-detects column names for ZINC ID and score to handle minor
    variations in the results_writer output format.
    """
    results = pd.read_csv(results_path)
    print(f"Vina results           : {len(results):,} rows  ({results_path})")
    print(f"  columns: {list(results.columns)}")

    # Normalise column names
    col_map = {}
    for col in results.columns:
        lc = col.lower()
        if lc in ("ligand_name", "ligand", "zinc_id", "name", "compound_id", "id"):
            col_map[col] = "zinc_id"
        elif lc in ("score", "docking_score", "affinity", "best_score", "energy"):
            col_map[col] = "docking_score"
        elif lc == "success":
            col_map[col] = "success"
    results = results.rename(columns=col_map)

    # Keep only successful docks
    if "success" in results.columns:
        n_before = len(results)
        results  = results[results["success"] == True]
        print(f"  After success filter   : {len(results):,}  (dropped {n_before - len(results):,} failures)")

    if "zinc_id" not in results.columns:
        raise KeyError(
            "Could not identify a ZINC ID column in Vina results. "
            f"Available columns: {list(results.columns)}"
        )
    if "docking_score" not in results.columns:
        raise KeyError(
            "Could not identify a docking score column in Vina results. "
            f"Available columns: {list(results.columns)}"
        )

    results = results[["zinc_id", "docking_score"]].dropna()
    results["zinc_id"] = results["zinc_id"].astype(str).str.replace(
        r"\.pdbqt$", "", regex=True
    )

    # Join SMILES from the top-hits CSV
    smiles_df = pd.read_csv(smiles_map_path, usecols=["zinc_id", "smiles"])
    smiles_df["zinc_id"] = smiles_df["zinc_id"].astype(str)

    merged = results.merge(smiles_df, on="zinc_id", how="inner")
    print(f"  After SMILES join      : {len(merged):,} compounds")

    # Convert raw Vina score → Ligand Efficiency: LE = (score / n_heavy_atoms) * 100
    merged["docking_score"] = merged.apply(
        lambda row: docking_score_to_le(row["docking_score"], row["smiles"]),
        axis=1,
    )
    n_before = len(merged)
    merged = merged.dropna(subset=["docking_score"]).reset_index(drop=True)
    if n_before - len(merged):
        print(f"  Dropped {n_before - len(merged):,} rows where LE could not be computed")

    print(f"  LE score range         : {merged['docking_score'].min():.4f} to "
          f"{merged['docking_score'].max():.4f}")

    return merged[["zinc_id", "smiles", "docking_score"]]


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, n_epochs, lr, device, save_path):
    model     = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()
    best_val  = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(X)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item() * len(X)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:>3}/{n_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"             + saved  (val={best_val:.4f})")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge round-2 docking results with original training data and retrain"
    )
    parser.add_argument("--original-data",   required=True,
                        help="Original training CSV")
    parser.add_argument("--new-docking",     required=True,
                        help="Vina results CSV from round-2 docking campaign")
    parser.add_argument("--smiles-map",      required=True,
                        help="top_1M_hits.csv for ZINC-ID → SMILES lookup")
    parser.add_argument("--orig-zinc-col",   default="zinc_id",
                        help="ZINC ID column name in original data (default: zinc_id)")
    parser.add_argument("--orig-smiles-col", default="SMILES",
                        help="SMILES column name in original data (default: SMILES)")
    parser.add_argument("--orig-score-col",  default="LEScore",
                        help="Score column name in original data (default: LEScore)")
    parser.add_argument("--output-model",    default=None,
                        help="Output path for model weights (default: auto-timestamped)")
    parser.add_argument("--output-scaler",   default=None,
                        help="Output path for scaler (default: auto-timestamped)")
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--batch",     type=int,   default=1024)
    parser.add_argument("--dropout",   type=float, default=0.3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split",type=float, default=0.1)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%d%b%Y").lower()
    Path("trained_models").mkdir(exist_ok=True)
    model_path  = args.output_model  or f"trained_models/model_round2_{stamp}.pt"
    scaler_path = args.output_scaler or f"trained_models/scaler_round2_{stamp}.pkl"

    # ── Load and merge ────────────────────────────────────────────────────────
    df_orig = load_original_training_data(
        args.original_data,
        zinc_col=args.orig_zinc_col,
        smiles_col=args.orig_smiles_col,
        score_col=args.orig_score_col,
    )
    df_new = load_vina_results(args.new_docking, args.smiles_map)

    df_all = pd.concat([df_orig, df_new], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["zinc_id"]).reset_index(drop=True)

    print(f"\nCombined dataset : {len(df_all):,} unique compounds")
    print(f"  Original       : {len(df_orig):,}")
    print(f"  New (round 2)  : {len(df_new):,}")
    print(f"  LE score range : {df_all['docking_score'].min():.4f} to "
          f"{df_all['docking_score'].max():.4f}  (kcal/mol per heavy atom × 100)\n")

    # ── Fingerprints ──────────────────────────────────────────────────────────
    print("Generating Morgan fingerprints ...")
    fps, valid_mask = generate_fingerprints(df_all["smiles"].tolist())
    scores = df_all.loc[valid_mask, "docking_score"].values.astype(np.float32)
    print(f"  {fps.shape[0]:,} valid / {len(df_all):,} total")

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler     = StandardScaler()
    fps_scaled = scaler.fit_transform(fps)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")

    # ── Dataset split ─────────────────────────────────────────────────────────
    dataset = MoleculeDataset(fps_scaled, scores)
    n_total = len(dataset)
    n_val   = int(n_total * args.val_split)
    n_test  = int(n_total * args.test_split)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Split: train={n_train:,}  val={n_val:,}  test={n_test:,}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = DockingScoreNet(input_dim=2048, hidden_dims=(1024, 512, 256), dropout=args.dropout)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(
        model, train_loader, val_loader,
        n_epochs=args.epochs, lr=args.lr,
        device=device, save_path=model_path,
    )

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    all_preds, all_true = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out   = model(X)
            test_loss += criterion(out, y).item() * len(X)
            all_preds.extend(out.squeeze(1).cpu().numpy())
            all_true.extend(y.squeeze(1).cpu().numpy())

    test_rmse = float(np.sqrt(test_loss / len(test_loader.dataset)))
    print(f"\nTest RMSE : {test_rmse:.4f} LE units")
    print(f"Model  saved : {model_path}")
    print(f"Scaler saved : {scaler_path}")


if __name__ == "__main__":
    main()
