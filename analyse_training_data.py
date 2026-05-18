"""
Training Data Chemical Space Analysis
=======================================
Comprehensive analysis of the initial docking dataset (~1.1M compounds).
Designed for large datasets -- fingerprint-heavy operations subsample
intelligently so the full 1.1M does not need to fit in RAM at once.

Plots and analyses generated:
  1.  Score distribution (full dataset)
  2.  Score distribution by decile
  3.  PCA of Morgan fingerprint space  (subsampled)
  4.  UMAP of Morgan fingerprint space (subsampled)
  5.  Molecular property distributions (MW, LogP, HBD, HBA, TPSA, RotBonds)
  6.  Lipinski Rule of 5 compliance
  7.  QED (drug-likeness) distribution
  8.  Score vs property correlation plots
  9.  Murcko scaffold diversity
  10. Internal diversity (mean pairwise Tanimoto on random sample)
  11. Nearest-neighbour Tanimoto distribution
  12. Top and bottom compound structure grids
  13. Summary statistics report (text file)

Dependencies:
    pip install rdkit pandas numpy matplotlib seaborn scikit-learn umap-learn

Usage:
    python analyse_training_data.py
"""

import os
import math
import warnings
import multiprocessing
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import List, Optional, Tuple

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem, Draw, Descriptors, rdFingerprintGenerator,
    QED as rdQED
)
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("umap-learn not installed -- UMAP plot skipped (pip install umap-learn)")
    UMAP_AVAILABLE = False


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

TRAINING_CSV     = "../qcrb_docking_data/normalised_scores.csv"
OUTPUT_DIR       = "output_old/"

ZINC_COL         = "zinc_name"
SMILES_COL       = "SMILES"
SCORE_COL        = "normalised_score"

# Subsampling sizes -- full 1.1M is too large for fingerprint matrix ops
N_FOR_PCA        = 100000    # random sample for PCA / UMAP
N_FOR_PROPS      = 200000    # random sample for property calculations
N_FOR_DIVERSITY  = 5000      # random sample for pairwise Tanimoto (O(n^2))
N_FOR_SCAFFOLD   = 50000     # random sample for scaffold analysis
N_TOP_STRUCT     = 40        # compounds to draw in structure grid
N_BOTTOM_STRUCT  = 40        # worst-scoring compounds to draw

RADIUS           = 2
N_BITS           = 2048
CHUNK_SIZE       = 50000     # rows per chunk for property calculation
N_WORKERS        = os.cpu_count()

PLOT_DPI         = 150
BG_DARK          = "#0d1117"
BG_MED           = "#161b22"
BG_BORDER        = "#30363d"
COLOURS = {
    "main":    "#2a9d8f",
    "accent":  "#e63946",
    "mid":     "#f4a261",
    "soft":    "#457b9d",
    "neutral": "#adb5bd",
}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _ax_style(ax):
    """Apply dark theme to a matplotlib axis."""
    ax.set_facecolor(BG_MED)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor(BG_BORDER)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def _save(fig, path):
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024
    print("  Saved: {}  ({:.0f} KB)".format(os.path.basename(path), size_kb))


def load_data(csv_path: str) -> pd.DataFrame:
    print("Loading '{}' ...".format(csv_path))
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=[SMILES_COL, SCORE_COL]).reset_index(drop=True)
    print("  {:,} compounds loaded".format(len(df)))
    print("  Score range : {:.4f}  to  {:.4f}".format(
        df[SCORE_COL].min(), df[SCORE_COL].max()))
    return df


# ------------------------------------------------------------------------------
# Property calculation (chunked to stay memory-safe at 1.1M)
# ------------------------------------------------------------------------------

def _calc_props_chunk(smiles_list: List[str]) -> pd.DataFrame:
    """Calculate properties for a list of SMILES strings."""
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            rows.append({
                "mw":       Descriptors.MolWt(mol),
                "logp":     Descriptors.MolLogP(mol),
                "hbd":      Descriptors.NumHDonors(mol),
                "hba":      Descriptors.NumHAcceptors(mol),
                "tpsa":     Descriptors.TPSA(mol),
                "rotbonds": Descriptors.NumRotatableBonds(mol),
                "rings":    Descriptors.RingCount(mol),
                "qed":      rdQED.qed(mol),
                "heavy":    mol.GetNumHeavyAtoms(),
                # Lipinski violations
                "lip_mw":   int(Descriptors.MolWt(mol) > 500),
                "lip_logp": int(Descriptors.MolLogP(mol) > 5),
                "lip_hbd":  int(Descriptors.NumHDonors(mol) > 5),
                "lip_hba":  int(Descriptors.NumHAcceptors(mol) > 10),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def calculate_properties(df: pd.DataFrame,
                          n_sample: int = N_FOR_PROPS) -> pd.DataFrame:
    """
    Calculate molecular properties for a subsample, chunked to limit RAM.
    Returns property DataFrame aligned with the input sample.
    """
    sample = df.sample(min(n_sample, len(df)), random_state=42)
    smiles = sample[SMILES_COL].tolist()
    scores = sample[SCORE_COL].tolist()

    print("Calculating properties for {:,} compounds ...".format(len(smiles)))
    chunks = [smiles[i: i + CHUNK_SIZE]
              for i in range(0, len(smiles), CHUNK_SIZE)]

    prop_frames = []
    for i, chunk in enumerate(chunks, start=1):
        prop_frames.append(_calc_props_chunk(chunk))
        if i % 5 == 0 or i == len(chunks):
            done = min(i * CHUNK_SIZE, len(smiles))
            print("  {:>7,} / {:,}".format(done, len(smiles)))

    props = pd.concat(prop_frames, ignore_index=True)

    # Align scores (drop rows where RDKit failed)
    # Re-calculate to align lengths properly
    scores_aligned = []
    for smi, score in zip(smiles, scores):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scores_aligned.append(score)
    props[SCORE_COL] = scores_aligned[:len(props)]

    print("  {:,} valid property rows".format(len(props)))
    return props


# ------------------------------------------------------------------------------
# Fingerprint generation (parallel)
# ------------------------------------------------------------------------------

def _fp_worker(args: Tuple) -> Tuple[List[np.ndarray], List[int]]:
    chunk, radius, n_bits = args
    gen  = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps  = []
    idxs = []
    for idx, smi in chunk:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(gen.GetFingerprintAsNumPy(mol).astype(np.float32))
        idxs.append(idx)
    return fps, idxs


def build_fp_matrix(df: pd.DataFrame,
                    n_sample: int,
                    label: str = "") -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate fingerprints for a random subsample of df.
    Returns (fp_matrix, subsampled_df).
    """
    sample = df.sample(min(n_sample, len(df)), random_state=42).reset_index(drop=True)
    pairs  = list(zip(sample.index.tolist(), sample[SMILES_COL].tolist()))

    n_proc    = min(N_WORKERS, len(pairs))
    cs        = math.ceil(len(pairs) / n_proc)
    chunks    = [pairs[i: i + cs] for i in range(0, len(pairs), cs)]
    pool_args = [(c, RADIUS, N_BITS) for c in chunks]

    print("Generating fingerprints{} ({:,} compounds, {} workers) ...".format(
        " for {}".format(label) if label else "", len(pairs), n_proc))

    with multiprocessing.Pool(processes=n_proc) as pool:
        results = pool.map(_fp_worker, pool_args)

    flat_fps = [fp  for fps, _    in results for fp  in fps]
    flat_idx = [idx for _,   idxs in results for idx in idxs]

    fp_matrix  = np.vstack(flat_fps)
    valid_df   = sample.loc[flat_idx].reset_index(drop=True)
    print("  {:,} valid fingerprints".format(len(fp_matrix)))
    return fp_matrix, valid_df


# ------------------------------------------------------------------------------
# Plot 1 & 2: Score distributions
# ------------------------------------------------------------------------------

def plot_score_distributions(df: pd.DataFrame, out_dir: str) -> None:
    print("\nPlotting score distributions ...")
    scores = df[SCORE_COL]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle(
        "Docking Score Distribution  (n={:,})".format(len(df)),
        color="white", fontsize=14, y=1.01)

    # 1. Full histogram
    ax = axes[0, 0]
    _ax_style(ax)
    ax.hist(scores, bins=150, color=COLOURS["main"],
            edgecolor="none", alpha=0.85)
    for pct, col, lbl in [(1, COLOURS["accent"], "Top 1%"),
                           (5, COLOURS["mid"],    "Top 5%"),
                           (10, COLOURS["soft"],   "Top 10%")]:
        ax.axvline(scores.quantile(pct / 100), color=col, lw=1.5,
                   linestyle="--", label=lbl)
    ax.set_xlabel("Docking Score (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title("Full Distribution")
    ax.legend(framealpha=0.3, labelcolor="white")

    # 2. Log-scale to show tail detail
    ax = axes[0, 1]
    _ax_style(ax)
    ax.hist(scores, bins=150, color=COLOURS["soft"],
            edgecolor="none", alpha=0.85, log=True)
    ax.set_xlabel("Docking Score (kcal/mol)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Full Distribution (log y)")

    # 3. Boxplot by decile
    ax = axes[1, 0]
    _ax_style(ax)
    df_tmp = df[[SCORE_COL]].copy()
    df_tmp["decile"] = pd.qcut(scores, 10,
                                labels=["D{}".format(i) for i in range(1, 11)])
    decile_data = [df_tmp[df_tmp["decile"] == d][SCORE_COL].values
                   for d in ["D{}".format(i) for i in range(1, 11)]]
    bp = ax.boxplot(decile_data, patch_artist=True,
                    medianprops=dict(color="white", lw=2))
    colours_grad = plt.cm.plasma_r(np.linspace(0.1, 0.9, 10))
    for patch, col in zip(bp["boxes"], colours_grad):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color(COLOURS["neutral"])
    ax.set_xticklabels(
        ["D{}".format(i) for i in range(1, 11)], color="white")
    ax.set_xlabel("Score Decile (D1=best)")
    ax.set_ylabel("Docking Score (kcal/mol)")
    ax.set_title("Score by Decile")

    # 4. Cumulative distribution
    ax = axes[1, 1]
    _ax_style(ax)
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, cdf, color=COLOURS["main"], lw=2)
    ax.set_xlabel("Docking Score (kcal/mol)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("Cumulative Distribution")
    ax.axhline(0.01, color=COLOURS["accent"],
               linestyle="--", lw=1, label="1%")
    ax.axhline(0.05, color=COLOURS["mid"],
               linestyle="--", lw=1, label="5%")
    ax.legend(framealpha=0.3, labelcolor="white")

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "01_score_distributions.png"))


# ------------------------------------------------------------------------------
# Plot 3: PCA
# ------------------------------------------------------------------------------

def plot_pca(fp_matrix: np.ndarray,
             df: pd.DataFrame,
             out_dir: str) -> np.ndarray:
    print("\nRunning PCA ...")
    scaler = StandardScaler(with_mean=False)
    fps_sc = scaler.fit_transform(fp_matrix)
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(fps_sc)
    var    = pca.explained_variance_ratio_ * 100
    print("  PC1={:.1f}%  PC2={:.1f}%".format(var[0], var[1]))

    scores = df[SCORE_COL].values
    norm   = plt.Normalize(vmin=np.percentile(scores, 1),
                           vmax=np.percentile(scores, 99))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG_DARK)

    # Coloured by score
    ax = axes[0]
    _ax_style(ax)
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=scores, cmap="plasma_r", norm=norm,
                    s=3, alpha=0.4, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Docking Score", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_xlabel("PC1 ({:.1f}% var)".format(var[0]))
    ax.set_ylabel("PC2 ({:.1f}% var)".format(var[1]))
    ax.set_title("PCA -- Coloured by Docking Score\n(n={:,})".format(
        len(df)))

    # Highlight top 1%
    ax = axes[1]
    _ax_style(ax)
    top1_mask = scores <= np.percentile(scores, 1)
    ax.scatter(coords[~top1_mask, 0], coords[~top1_mask, 1],
               c=COLOURS["neutral"], s=2, alpha=0.2,
               linewidths=0, label="Rest")
    ax.scatter(coords[top1_mask, 0], coords[top1_mask, 1],
               c=COLOURS["accent"], s=8, alpha=0.8,
               linewidths=0, label="Top 1% ({:,})".format(top1_mask.sum()))
    ax.set_xlabel("PC1 ({:.1f}% var)".format(var[0]))
    ax.set_ylabel("PC2 ({:.1f}% var)".format(var[1]))
    ax.set_title("PCA -- Top 1% Highlighted")
    leg = ax.legend(framealpha=0.3, labelcolor="white")

    plt.suptitle("PCA of Morgan Fingerprints  (subsample n={:,})".format(
        len(df)), color="white", fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "03_pca.png"))
    return coords


# ------------------------------------------------------------------------------
# Plot 4: UMAP
# ------------------------------------------------------------------------------

def plot_umap(fp_matrix: np.ndarray,
              df: pd.DataFrame,
              out_dir: str) -> None:
    if not UMAP_AVAILABLE:
        return
    print("\nRunning UMAP (may take several minutes) ...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                         metric="jaccard", random_state=42, low_memory=True)
    coords  = reducer.fit_transform(fp_matrix)

    scores   = df[SCORE_COL].values
    norm     = plt.Normalize(vmin=np.percentile(scores, 1),
                             vmax=np.percentile(scores, 99))
    top1_mask = scores <= np.percentile(scores, 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG_DARK)

    ax = axes[0]
    _ax_style(ax)
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=scores, cmap="plasma_r", norm=norm,
                    s=3, alpha=0.4, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Docking Score", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP -- Coloured by Score  (Jaccard metric)")

    ax = axes[1]
    _ax_style(ax)
    ax.scatter(coords[~top1_mask, 0], coords[~top1_mask, 1],
               c=COLOURS["neutral"], s=2, alpha=0.15, linewidths=0,
               label="Rest")
    ax.scatter(coords[top1_mask, 0], coords[top1_mask, 1],
               c=COLOURS["accent"], s=8, alpha=0.85, linewidths=0,
               label="Top 1%")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP -- Top 1% Highlighted")
    ax.legend(framealpha=0.3, labelcolor="white")

    plt.suptitle("UMAP of Morgan Fingerprints  (subsample n={:,})".format(
        len(df)), color="white", fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "04_umap.png"))


# ------------------------------------------------------------------------------
# Plot 5: Molecular properties
# ------------------------------------------------------------------------------

def plot_properties(props: pd.DataFrame, out_dir: str) -> None:
    print("\nPlotting property distributions ...")

    prop_defs = [
        ("mw",       "Molecular Weight (Da)",    500,  "Lipinski MW < 500"),
        ("logp",     "LogP",                      5,   "Lipinski LogP < 5"),
        ("hbd",      "H-Bond Donors",             5,   "Lipinski HBD < 5"),
        ("hba",      "H-Bond Acceptors",         10,   "Lipinski HBA < 10"),
        ("tpsa",     "TPSA (A^2)",               140,  "Oral bioavail. TPSA < 140"),
        ("rotbonds", "Rotatable Bonds",           10,   "Veber RotBonds < 10"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle(
        "Molecular Property Distributions  (n={:,})".format(len(props)),
        color="white", fontsize=14)

    for ax, (col, label, limit, limit_label) in zip(axes.flat, prop_defs):
        _ax_style(ax)
        vals = props[col].dropna()

        # Colour bars by whether they exceed the rule-of-5 limit
        n_bins = 60
        counts, bin_edges = np.histogram(vals, bins=n_bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_colours = [COLOURS["accent"] if c > limit else COLOURS["main"]
                       for c in bin_centres]
        ax.bar(bin_centres, counts, width=(bin_edges[1] - bin_edges[0]),
               color=bar_colours, edgecolor="none", alpha=0.85)
        ax.axvline(limit, color="white", lw=1.5,
                   linestyle="--", label=limit_label)

        pct_ok = 100 * (vals <= limit).mean()
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title("{}\n{:.1f}% within limit".format(label, pct_ok))
        ax.legend(framealpha=0.2, labelcolor="white", fontsize=7)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "05_property_distributions.png"))


# ------------------------------------------------------------------------------
# Plot 6: Lipinski compliance
# ------------------------------------------------------------------------------

def plot_lipinski(props: pd.DataFrame, out_dir: str) -> None:
    print("\nPlotting Lipinski compliance ...")

    props = props.copy()
    props["n_violations"] = (props["lip_mw"] + props["lip_logp"] +
                              props["lip_hbd"] + props["lip_hba"])

    labels  = ["0 violations\n(fully compliant)",
               "1 violation", "2 violations",
               "3 violations", "4 violations\n(fully non-compliant)"]
    counts  = [( props["n_violations"] == i).sum() for i in range(5)]
    colours = ["#2a9d8f", "#57cc99", "#f4a261", "#e76f51", "#e63946"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG_DARK)

    # Pie
    ax = axes[0]
    ax.set_facecolor(BG_DARK)
    wedges, texts, autos = ax.pie(
        counts, labels=None, autopct="%1.1f%%",
        colors=colours, startangle=140, pctdistance=0.8)
    for at in autos:
        at.set_color("white")
        at.set_fontsize(9)
    ax.legend(wedges, labels, loc="lower left",
              bbox_to_anchor=(-0.1, -0.15),
              labelcolor="white", framealpha=0.2, fontsize=9)
    ax.set_title("Lipinski Rule-of-5 Compliance\n(n={:,})".format(len(props)),
                 color="white")

    # Bar -- per individual rule
    ax = axes[1]
    _ax_style(ax)
    rules  = ["MW > 500", "LogP > 5", "HBD > 5", "HBA > 10"]
    viol   = [props["lip_mw"].sum(), props["lip_logp"].sum(),
              props["lip_hbd"].sum(), props["lip_hba"].sum()]
    pcts   = [100 * v / len(props) for v in viol]
    bars   = ax.barh(rules, pcts,
                     color=[COLOURS["accent"], COLOURS["mid"],
                             COLOURS["soft"], COLOURS["main"]],
                     edgecolor="none")
    for bar, pct in zip(bars, pcts):
        ax.text(pct + 0.3, bar.get_y() + bar.get_height() / 2,
                "{:.1f}%".format(pct), va="center", color="white", fontsize=10)
    ax.set_xlabel("% of compounds violating rule")
    ax.set_title("Individual Rule Violations")
    ax.set_xlim(0, max(pcts) * 1.2)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "06_lipinski_compliance.png"))


# ------------------------------------------------------------------------------
# Plot 7: QED distribution
# ------------------------------------------------------------------------------

def plot_qed(props: pd.DataFrame, out_dir: str) -> None:
    print("\nPlotting QED distribution ...")

    qed = props["qed"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG_DARK)

    ax = axes[0]
    _ax_style(ax)
    ax.hist(qed, bins=80, color=COLOURS["main"],
            edgecolor="none", alpha=0.85)
    ax.axvline(qed.median(), color=COLOURS["accent"],
               lw=2, linestyle="--",
               label="Median: {:.3f}".format(qed.median()))
    ax.axvline(0.5, color="white", lw=1.5, linestyle=":",
               label="QED = 0.5 (guideline)")
    ax.set_xlabel("QED Score (0=least, 1=most drug-like)")
    ax.set_ylabel("Count")
    ax.set_title("QED Distribution\n(n={:,})".format(len(qed)))
    ax.legend(framealpha=0.3, labelcolor="white")

    # QED vs docking score scatter
    ax = axes[1]
    _ax_style(ax)
    mask = props["qed"].notna() & props[SCORE_COL].notna()
    ax.hexbin(props.loc[mask, "qed"],
              props.loc[mask, SCORE_COL],
              gridsize=50, cmap="plasma_r",
              mincnt=1, linewidths=0)
    ax.set_xlabel("QED Score")
    ax.set_ylabel("Docking Score (kcal/mol)")
    ax.set_title("QED vs Docking Score")

    pct_above = 100 * (qed >= 0.5).mean()
    fig.suptitle("QED Drug-likeness  --  {:.1f}% of compounds QED >= 0.5".format(
        pct_above), color="white", fontsize=12)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "07_qed.png"))


# ------------------------------------------------------------------------------
# Plot 8: Score vs property correlations
# ------------------------------------------------------------------------------

def plot_score_correlations(props: pd.DataFrame, out_dir: str) -> None:
    print("\nPlotting score-property correlations ...")

    prop_cols = ["mw", "logp", "hbd", "hba", "tpsa", "qed"]
    prop_lbls = ["MW", "LogP", "HBD", "HBA", "TPSA", "QED"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle("Docking Score vs Molecular Properties",
                 color="white", fontsize=14)

    for ax, col, lbl in zip(axes.flat, prop_cols, prop_lbls):
        _ax_style(ax)
        mask = props[col].notna() & props[SCORE_COL].notna()
        ax.hexbin(props.loc[mask, col],
                  props.loc[mask, SCORE_COL],
                  gridsize=50, cmap="YlOrRd",
                  mincnt=1, linewidths=0)
        corr = props.loc[mask, [col, SCORE_COL]].corr().iloc[0, 1]
        ax.set_xlabel(lbl)
        ax.set_ylabel("Docking Score")
        ax.set_title("{}  (r={:.3f})".format(lbl, corr))

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "08_score_correlations.png"))


# ------------------------------------------------------------------------------
# Plot 9: Scaffold diversity
# ------------------------------------------------------------------------------

def plot_scaffolds(df: pd.DataFrame, out_dir: str) -> None:
    print("\nAnalysing Murcko scaffolds ({:,} sample) ...".format(
        N_FOR_SCAFFOLD))
    sample = df.sample(min(N_FOR_SCAFFOLD, len(df)), random_state=42)

    scaffolds = []
    for smi in sample[SMILES_COL]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds.append("INVALID")
            continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False)
            scaffolds.append(sc if sc else "NO_SCAFFOLD")
        except Exception:
            scaffolds.append("ERROR")

    sample = sample.copy()
    sample["scaffold"]  = scaffolds
    counts              = sample["scaffold"].value_counts()
    n_unique            = (counts.index != "INVALID").sum()
    singleton_pct       = 100 * (counts == 1).sum() / len(counts)

    print("  Unique scaffolds  : {:,}".format(n_unique))
    print("  Singleton scaffolds: {:.1f}%".format(singleton_pct))

    top15     = counts.head(15)
    other_cnt = counts.iloc[15:].sum()
    if other_cnt:
        top15["Other ({:,})".format(len(counts) - 15)] = other_cnt

    colours = plt.cm.tab20(np.linspace(0, 1, len(top15)))

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor(BG_DARK)

    ax = axes[0]
    ax.set_facecolor(BG_DARK)
    wedges, _, autos = ax.pie(
        top15.values, labels=None, autopct="%1.1f%%",
        colors=colours, startangle=140, pctdistance=0.82)
    for at in autos:
        at.set_color("white")
        at.set_fontsize(7)
    ax.legend(wedges,
              [s[:38] + "..." if len(s) > 38 else s
               for s in top15.index],
              loc="center left", bbox_to_anchor=(1, 0.5),
              labelcolor="white", framealpha=0.15, fontsize=7)
    ax.set_title(
        "Murcko Scaffold Distribution\n"
        "{:,} unique scaffolds  |  {:.1f}% singletons".format(
            n_unique, singleton_pct),
        color="white", fontsize=11)

    ax = axes[1]
    _ax_style(ax)
    top20 = counts.head(20)
    ax.barh(range(len(top20)), top20.values,
            color=plt.cm.plasma_r(np.linspace(0.1, 0.9, len(top20))))
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(
        [s[:40] + "..." if len(s) > 40 else s for s in top20.index],
        fontsize=7, color="white")
    ax.set_xlabel("Number of compounds")
    ax.set_title("Top 20 Most Common Scaffolds")
    ax.invert_yaxis()

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "09_scaffold_diversity.png"))


# ------------------------------------------------------------------------------
# Plot 10: Internal diversity (pairwise Tanimoto)
# ------------------------------------------------------------------------------

def plot_internal_diversity(df: pd.DataFrame, out_dir: str) -> None:
    print("\nCalculating internal diversity ({:,} sample) ...".format(
        N_FOR_DIVERSITY))
    sample = df.sample(min(N_FOR_DIVERSITY, len(df)), random_state=42)
    gen    = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=N_BITS)

    rdkit_fps = []
    for smi in sample[SMILES_COL]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            rdkit_fps.append(gen.GetFingerprint(mol))

    n        = len(rdkit_fps)
    sim_vals = []
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(rdkit_fps[i], rdkit_fps[i+1:])
        sim_vals.extend(sims)

    sim_arr      = np.array(sim_vals, dtype=np.float32)
    mean_sim     = sim_arr.mean()
    mean_dist    = 1 - mean_sim
    print("  Mean pairwise Tanimoto similarity : {:.4f}".format(mean_sim))
    print("  Mean pairwise Tanimoto distance   : {:.4f}".format(mean_dist))
    print("  (Higher distance = more diverse dataset)")

    # Nearest-neighbour similarity (max per compound excl. self)
    nn_sims = []
    for i in range(n):
        all_sims = DataStructs.BulkTanimotoSimilarity(rdkit_fps[i], rdkit_fps)
        all_sims[i] = 0.0   # exclude self
        nn_sims.append(max(all_sims))
    nn_arr = np.array(nn_sims, dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_DARK)

    ax = axes[0]
    _ax_style(ax)
    ax.hist(sim_vals, bins=80, color=COLOURS["main"],
            edgecolor="none", alpha=0.85)
    ax.axvline(mean_sim, color=COLOURS["accent"], lw=2,
               linestyle="--",
               label="Mean: {:.3f}".format(mean_sim))
    ax.set_xlabel("Pairwise Tanimoto Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Pairwise Tanimoto Distribution\n"
                 "Mean distance = {:.3f}  (n={:,} pairs)".format(
                     mean_dist, len(sim_vals)))
    ax.legend(framealpha=0.3, labelcolor="white")

    ax = axes[1]
    _ax_style(ax)
    ax.hist(nn_sims, bins=60, color=COLOURS["soft"],
            edgecolor="none", alpha=0.85)
    ax.axvline(nn_arr.mean(), color=COLOURS["accent"], lw=2,
               linestyle="--",
               label="Mean NN: {:.3f}".format(nn_arr.mean()))
    ax.set_xlabel("Nearest-Neighbour Tanimoto Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Nearest-Neighbour Similarity\n"
                 "(low = isolated / unique compounds)")
    ax.legend(framealpha=0.3, labelcolor="white")

    plt.suptitle("Dataset Diversity  (subsample n={:,})".format(n),
                 color="white", fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "10_diversity.png"))

    return mean_sim, mean_dist


# ------------------------------------------------------------------------------
# Plot 11 & 12: Structure grids
# ------------------------------------------------------------------------------

def plot_structure_grids(df: pd.DataFrame, out_dir: str) -> None:
    print("\nDrawing structure grids ...")
    df_sorted = df.sort_values(SCORE_COL)

    for label, subset, fname in [
        ("Top {} Best Scoring Compounds".format(N_TOP_STRUCT),
         df_sorted.head(N_TOP_STRUCT),
         "11_top_structures.png"),
        ("Bottom {} Worst Scoring Compounds".format(N_BOTTOM_STRUCT),
         df_sorted.tail(N_BOTTOM_STRUCT),
         "12_bottom_structures.png"),
    ]:
        mols, legends = [], []
        for _, row in subset.iterrows():
            mol = Chem.MolFromSmiles(row[SMILES_COL])
            if mol is None:
                continue
            mols.append(mol)
            legends.append("{}\n{:.3f}".format(
                str(row[ZINC_COL])[:12], row[SCORE_COL]))

        img = Draw.MolsToGridImage(
            mols, molsPerRow=8, subImgSize=(300, 250),
            legends=legends, returnPNG=False)
        img.save(os.path.join(out_dir, fname))
        size_kb = os.path.getsize(
            os.path.join(out_dir, fname)) / 1024
        print("  Saved: {}  ({:.0f} KB)".format(fname, size_kb))


# ------------------------------------------------------------------------------
# Summary statistics text file
# ------------------------------------------------------------------------------

def write_summary(df: pd.DataFrame,
                  props: pd.DataFrame,
                  mean_sim: float,
                  mean_dist: float,
                  out_dir: str) -> None:
    scores = df[SCORE_COL]
    lines  = [
        "Training Data Summary",
        "=" * 60,
        "",
        "Dataset",
        "-" * 40,
        "  Total compounds         : {:,}".format(len(df)),
        "",
        "Docking Scores",
        "-" * 40,
        "  Min (best)              : {:.4f}".format(scores.min()),
        "  Max (worst)             : {:.4f}".format(scores.max()),
        "  Mean                    : {:.4f}".format(scores.mean()),
        "  Std                     : {:.4f}".format(scores.std()),
        "  Median                  : {:.4f}".format(scores.median()),
        "  Top  1% threshold       : {:.4f}  ({:,} cpds)".format(
            scores.quantile(0.01),
            (scores <= scores.quantile(0.01)).sum()),
        "  Top  5% threshold       : {:.4f}  ({:,} cpds)".format(
            scores.quantile(0.05),
            (scores <= scores.quantile(0.05)).sum()),
        "  Top 10% threshold       : {:.4f}  ({:,} cpds)".format(
            scores.quantile(0.10),
            (scores <= scores.quantile(0.10)).sum()),
        "",
        "Molecular Properties  (sample n={:,})".format(len(props)),
        "-" * 40,
        "  MW   mean / median      : {:.1f} / {:.1f} Da".format(
            props.mw.mean(), props.mw.median()),
        "  LogP mean / median      : {:.2f} / {:.2f}".format(
            props.logp.mean(), props.logp.median()),
        "  HBD  mean               : {:.2f}".format(props.hbd.mean()),
        "  HBA  mean               : {:.2f}".format(props.hba.mean()),
        "  TPSA mean               : {:.1f} A^2".format(props.tpsa.mean()),
        "  QED  mean               : {:.3f}".format(props.qed.mean()),
        "",
        "Lipinski Compliance",
        "-" * 40,
        "  0 violations (compliant): {:.1f}%".format(
            100 * (props.n_violations == 0).mean()
            if "n_violations" in props else float("nan")),
        "  MW > 500                : {:.1f}%".format(
            100 * props.lip_mw.mean()),
        "  LogP > 5                : {:.1f}%".format(
            100 * props.lip_logp.mean()),
        "  HBD > 5                 : {:.1f}%".format(
            100 * props.lip_hbd.mean()),
        "  HBA > 10                : {:.1f}%".format(
            100 * props.lip_hba.mean()),
        "",
        "Diversity",
        "-" * 40,
        "  Mean pairwise Tanimoto similarity : {:.4f}".format(mean_sim),
        "  Mean pairwise Tanimoto distance   : {:.4f}".format(mean_dist),
        "  (1.0 = identical, 0.0 = maximally diverse)",
        "",
        "Output directory : {}".format(out_dir),
    ]

    # Add n_violations if available
    if "n_violations" not in props.columns:
        props = props.copy()
        props["n_violations"] = (props["lip_mw"] + props["lip_logp"] +
                                  props["lip_hbd"] + props["lip_hba"])
    lines[lines.index("  0 violations (compliant): {:.1f}%".format(
        100 * (props.n_violations == 0).mean()
        if "n_violations" in props else float("nan")))] = \
        "  0 violations (compliant): {:.1f}%".format(
            100 * (props["n_violations"] == 0).mean())

    out_path = os.path.join(out_dir, "00_summary.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    print("\nSummary saved: {}".format(out_path))


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    multiprocessing.freeze_support()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.style.use("dark_background")

    # Load full dataset
    df = load_data(TRAINING_CSV)

    # Score distributions (uses full dataset)
    plot_score_distributions(df, OUTPUT_DIR)

    # Properties (chunked, subsampled)
    props = calculate_properties(df, n_sample=N_FOR_PROPS)
    plot_properties(props, OUTPUT_DIR)
    plot_lipinski(props, OUTPUT_DIR)
    plot_qed(props, OUTPUT_DIR)
    plot_score_correlations(props, OUTPUT_DIR)

    # Fingerprint-based analyses (subsampled)
    fp_matrix, fp_df = build_fp_matrix(df, N_FOR_PCA, label="PCA/UMAP")
    plot_pca(fp_matrix, fp_df, OUTPUT_DIR)
    plot_umap(fp_matrix, fp_df, OUTPUT_DIR)

    # Scaffold diversity
    plot_scaffolds(df, OUTPUT_DIR)

    # Internal diversity
    mean_sim, mean_dist = plot_internal_diversity(df, OUTPUT_DIR)

    # Structure grids
    plot_structure_grids(df, OUTPUT_DIR)

    # Summary
    write_summary(df, props, mean_sim, mean_dist, OUTPUT_DIR)

    print("\n" + "=" * 55)
    print("Analysis complete.  Outputs in '{}'".format(OUTPUT_DIR))
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        print("  {:45s}  {:>7.1f} KB".format(
            fname, os.path.getsize(fpath) / 1024))


if __name__ == "__main__":
    main()
