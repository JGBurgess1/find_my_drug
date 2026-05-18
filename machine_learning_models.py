"""
compare_models.py
=================
Compares scikit-learn and PyTorch models for predicting ligand efficiency
from Morgan fingerprints on a dataset of 1.1 million compounds.

Models compared:
  - scikit-learn:
    - LinearRegression
    - Ridge
    - Lasso
    - ElasticNet
    - SVR (Support Vector Regression)
    - MLPRegressor (Multi-layer Perceptron)
    - RandomForestRegressor
    - GradientBoostingRegressor
  - PyTorch:
    - Custom fully-connected neural network

Metrics:
  - R^2 score (coefficient of determination)
  - Mean squared error (MSE)
  - Root mean squared error (RMSE)
  - Mean absolute error (MAE)
  - Explained variance score

Usage:
  python compare_models.py --train_size 0.8 --n_jobs -1

Note: Adjust the `train_size` argument to control the train/validation split.
      Set `n_jobs` to the number of CPU cores to use for parallel training
      (-1 uses all available cores).
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

CSV_PATH        = "LE_Scores.csv"
SMILES_COL      = "SMILES"
LE_COL          = "LEScore"
FP_RADIUS       = 2
FP_SIZE         = 2048
TRAIN_SIZE      = 0.8       # fraction of data to use for training
BATCH_SIZE      = 1024      # for PyTorch dataloader
N_EPOCHS        = 100       # for PyTorch training
LR              = 1e-3      # learning rate for PyTorch optimizer
HIDDEN_DIMS     = [1024, 512, 256, 128]   # for PyTorch model

# Scikit-learn model configurations
# Adjust as needed (e.g. regularization strengths, tree depths)
SKL_MODELS = {
    "Linear":           LinearRegression(),
    "Ridge":            Ridge(alpha=1.0),
    "Lasso":            Lasso(alpha=0.1),
    "ElasticNet":       ElasticNet(alpha=0.1, l1_ratio=0.5),
    "SVR":              SVR(C=1.0, epsilon=0.1),
    "MLP":              MLPRegressor(hidden_layer_sizes=HIDDEN_DIMS,
                                     max_iter=500, batch_size=BATCH_SIZE,
                                     early_stopping=True),
    "RandomForest":     RandomForestRegressor(n_estimators=100),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100),
}


# ------------------------------------------------------------------------------
# Fingerprint generation
# ------------------------------------------------------------------------------

def get_fingerprints(smiles_list, radius=FP_RADIUS, nBits=FP_SIZE):
    fps = []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = generator.GetFingerprintAsNumPy(mol).astype(np.float32)
        if fp is not None:
            fps.append(fp)
    return np.array(fps, dtype=np.float32)


# ------------------------------------------------------------------------------
# PyTorch dataset and model
# ------------------------------------------------------------------------------

class MorganDataset(Dataset):
    def __init__(self, fps, labels):
        self.fps    = fps
        self.labels = labels

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        return self.fps[idx], self.labels[idx]


class MorganNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super(MorganNet, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ------------------------------------------------------------------------------
# Training and evaluation
# ------------------------------------------------------------------------------

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for fps, labels in train_loader:
        fps, labels = fps.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(fps)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(fps)
    return train_loss / len(train_loader.dataset)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    preds = []
    true = []
    with torch.no_grad():
        for fps, labels in val_loader:
            fps, labels = fps.to(device), labels.to(device)
            outputs = model(fps)
            val_loss += criterion(outputs, labels.unsqueeze(1)).item() * len(fps)
            preds.append(outputs.cpu().numpy())
            true.append(labels.cpu().numpy())
    preds = np.concatenate(preds).flatten()
    true = np.concatenate(true).flatten()
    return val_loss / len(val_loader.dataset), preds, true


def train_torch_model(model, train_loader, val_loader, criterion, optimizer,
                      n_epochs, device, early_stop_rounds=10):
    best_loss = float("inf")
    best_model_dict = model.state_dict()
    no_improve_count = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, preds, true = evaluate(model, val_loader, criterion, device)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_dict = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        if no_improve_count >= early_stop_rounds:
            print(f"Early stopping at epoch {epoch} due to no improvement in "
                  f"validation loss for {early_stop_rounds} rounds.")
            break

    model.load_state_dict(best_model_dict)
    return model, preds, true


# ------------------------------------------------------------------------------
# Metrics and reporting
# ------------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    evs  = explained_variance_score(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    pcc  = pearsonr(y_true, y_pred)[0]  # Pearson correlation coefficient
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "EVS": evs, "R2": r2, "PCC": pcc}


def evaluate_sklearn_models(X_train, X_val, y_train, y_val, models, n_jobs=-1):
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        results[name] = calculate_metrics(y_val, y_pred)
        print(results[name])
    return pd.DataFrame(results).T


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main(args):
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    print(f"Loading data from '{CSV_PATH}' ...")
    df = pd.read_csv(CSV_PATH)
    df = df.iloc[:10000, :]
    print(f"  Loaded {len(df):,} rows")

    # Generate fingerprints
    print(f"\nGenerating {FP_SIZE}-bit radius {FP_RADIUS} Morgan fingerprints ...")
    fps = get_fingerprints(df[SMILES_COL], radius=FP_RADIUS, nBits=FP_SIZE)
    labels = df[LE_COL].to_numpy()
    print(f"  Generated {len(fps):,} fingerprints")

    # Scale the labels (ligand efficiencies)
    scaler = StandardScaler()
    labels_scaled = scaler.fit_transform(labels.reshape(-1, 1)).flatten()

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        fps, labels_scaled, train_size=args.train_size, random_state=42)

    print(f"\nTrain size: {len(X_train):,}   Validation size: {len(X_val):,}")

    # Train scikit-learn models
    print("\nTraining scikit-learn models ...")
    skl_results = evaluate_sklearn_models(
        X_train, X_val, y_train, y_val, SKL_MODELS, n_jobs=args.n_jobs)
    print("\nScikit-learn results:\n")
    print(skl_results)

    # Train PyTorch model
    print("\nTraining PyTorch model ...")
    train_dataset = MorganDataset(X_train, y_train)
    val_dataset   = MorganDataset(X_val, y_val)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MorganNet(FP_SIZE, HIDDEN_DIMS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model, preds, true = train_torch_model(
        model, train_loader, val_loader, criterion, optimizer, N_EPOCHS, device)

    torch_metrics = calculate_metrics(true, preds.flatten())
    torch_results = pd.DataFrame([torch_metrics], index=["PyTorch"])
    print("\nPyTorch results:\n")
    print(torch_results)

    # Combined results
    print("\nCombined results:\n")
    results = pd.concat([skl_results, torch_results])
    print(results)

    print("\nBest model by RMSE:")
    print(results.sort_values("RMSE").head(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare ML models for predicting ligand efficiency from "
                    "Morgan fingerprints")
    parser.add_argument("--train_size", type=float, default=TRAIN_SIZE,
                        help="Fraction of data to use for training")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of CPU cores to use for scikit-learn models "
                             "(-1 = all available cores)")
    args = parser.parse_args()
    main(args)