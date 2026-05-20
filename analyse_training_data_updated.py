"""
Training Data Chemical Space Analysis
=======================================
Comprehensive analysis of the initial docking dataset (~1.1M compounds).

Output format:
  SVG  -- vector graphics, infinitely scalable, insert directly into Word
           (Insert -> Pictures -> SVG in Word 2016+)
  PNG  -- 300 DPI raster fallback for older Word / PowerPoint

All plots use a clean white background suitable for publication /
Word documents.  No dark theme.

Plots generated:
  01_score_distributions.svg/png
  02_pca.svg/png
  03_umap.svg/png
  04_property_distributions.svg/png
  05_lipinski_compliance.svg/png
  06_qed.svg/png
  07_score_correlations.svg/png
  08_scaffold_diversity.svg/png
  09_diversity.svg/png
  10_top_structures.png          (raster only -- molecule images)
  11_bottom_structures.png
  00_summary.txt

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
import matplotlib.ticker as ticker
import seaborn as sns
from typing import List, Optional, Tuple

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem, Draw, Descriptors, rdFingerprintGenerator,
    QED as rdQED,
)
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("umap-learn not installed -- UMAP plot skipped")
    UMAP_AVAILABLE = False


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

TRAINING_CSV     = "./Inference_1_Top_1M_Hits/top_1M_hits.csv"
OUTPUT_DIR       = "./Inference_1_Top_1M_Hits/"

ZINC_COL         = "zinc_id"
SMILES_COL      = "smiles"
SCORE_COL       = "predicted_docking_score"

N_FOR_PCA       = 500000
N_FOR_PROPS     = 500000
N_FOR_DIVERSITY = 50000
N_FOR_SCAFFOLD  = 100000
N_TOP_STRUCT    = 50
N_BOTTOM_STRUCT = 50

RADIUS          = 2
N_BITS          = 2048
CHUNK_SIZE      = 50000
N_WORKERS       = os.cpu_count()

# Save both SVG and PNG
SAVE_SVG        = False
SAVE_PNG        = True
PNG_DPI         = 300     # publication quality

# ------------------------------------------------------------------------------
# Clean white colour palette
# ------------------------------------------------------------------------------

# All colours chosen to be clearly visible on a white background
C = {
    "teal":    "#00897B",   # main series
    "red":     "#E53935",   # accent / top compounds
    "amber":   "#FB8C00",   # secondary accent
    "blue":    "#1E88E5",   # third colour
    "purple":  "#8E24AA",   # fourth colour
    "grey":    "#757575",   # neutral
    "lgrey":   "#EEEEEE",   # axis backgrounds / fills
    "border":  "#BDBDBD",   # axis spines / grid
    "text":    "#212121",   # primary text
    "subtext": "#616161",   # secondary text
}

# Matplotlib style -- clean, white, publication-ready
plt.rcParams.update({
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
    "axes.edgecolor":        C["border"],
    "axes.labelcolor":       C["text"],
    "axes.labelsize":        11,
    "axes.titlesize":        12,
    "axes.titleweight":      "bold",
    "axes.titlecolor":       C["text"],
    "axes.grid":             True,
    "axes.grid.which":       "major",
    "grid.color":            C["lgrey"],
    "grid.linewidth":        0.8,
    "xtick.color":           C["subtext"],
    "ytick.color":           C["subtext"],
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.framealpha":     0.9,
    "legend.edgecolor":      C["border"],
    "legend.fontsize":       9,
    "figure.titlesize":      13,
    "figure.titleweight":    "bold",
    "font.family":           "sans-serif",
    "text.color":            C["text"],
    "savefig.facecolor":     "white",
    "savefig.edgecolor":     "white",
})


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _ax_clean(ax, grid: bool = True) -> None:
    """Apply clean white-background styling to an axis."""
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor(C["border"])
        spine.set_linewidth(0.8)
    if grid:
        ax.grid(True, color=C["lgrey"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def _save(fig, stem: str) -> None:
    """Save figure as SVG and/or PNG to OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.patch.set_facecolor("white")

    if SAVE_SVG:
        svg_path = os.path.join(OUTPUT_DIR, stem + ".svg")
        fig.savefig(svg_path, format="svg",
                    bbox_inches="tight", facecolor="white")
        size_kb = os.path.getsize(svg_path) / 1024
        print("  SVG: {}  ({:.0f} KB)".format(stem + ".svg", size_kb))

    if SAVE_PNG:
        png_path = os.path.join(OUTPUT_DIR, stem + ".png")
        fig.savefig(png_path, format="png", dpi=PNG_DPI,
                    bbox_inches="tight", facecolor="white")
        size_kb = os.path.getsize(png_path) / 1024
        print("  PNG: {}  ({:.0f} KB)".format(stem + ".png", size_kb))

    plt.close(fig)


def load_data(csv_path: str) -> pd.DataFrame:
    print("Loading '{}' ...".format(csv_path))
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=[SMILES_COL, SCORE_COL]).reset_index(drop=True)
    print("  {:,} compounds loaded".format(len(df)))
    print("  Score range : {:.4f}  to  {:.4f}".format(
        df[SCORE_COL].min(), df[SCORE_COL].max()))
    return df


# ------------------------------------------------------------------------------
# Property calculation
# ------------------------------------------------------------------------------

def _calc_props_chunk(smiles_list: List[str]) -> pd.DataFrame:
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
                "qed":      rdQED.qed(mol),
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
    sample = df.sample(min(n_sample, len(df)), random_state=42)
    smiles = sample[SMILES_COL].tolist()
    scores = sample[SCORE_COL].tolist()

    print("Calculating properties for {:,} compounds ...".format(len(smiles)))
    chunks = [smiles[i: i + CHUNK_SIZE]
              for i in range(0, len(smiles), CHUNK_SIZE)]
    frames = []
    for i, chunk in enumerate(chunks, start=1):
        frames.append(_calc_props_chunk(chunk))
        if i % 5 == 0 or i == len(chunks):
            print("  {:>7,} / {:,}".format(
                min(i * CHUNK_SIZE, len(smiles)), len(smiles)))

    props = pd.concat(frames, ignore_index=True)

    # Re-align scores
    scores_aligned = []
    for smi, score in zip(smiles, scores):
        if Chem.MolFromSmiles(smi) is not None:
            scores_aligned.append(score)
    props[SCORE_COL] = scores_aligned[:len(props)]
    props["n_violations"] = (props["lip_mw"] + props["lip_logp"] +
                              props["lip_hbd"] + props["lip_hba"])
    print("  {:,} valid property rows".format(len(props)))
    return props


# ------------------------------------------------------------------------------
# Fingerprint generation
# ------------------------------------------------------------------------------

def _fp_worker(args: Tuple) -> Tuple[List[np.ndarray], List[int]]:
    chunk, radius, n_bits = args
    gen  = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, idxs = [], []
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
    sample = df.sample(min(n_sample, len(df)), random_state=42).reset_index(drop=True)
    pairs  = list(zip(sample.index.tolist(), sample[SMILES_COL].tolist()))
    n_proc = min(N_WORKERS, len(pairs))
    cs     = math.ceil(len(pairs) / n_proc)
    chunks = [pairs[i: i + cs] for i in range(0, len(pairs), cs)]

    print("Generating fingerprints{} ({:,} cpds) ...".format(
        " for {}".format(label) if label else "", len(pairs)))

    with multiprocessing.Pool(processes=n_proc) as pool:
        results = pool.map(_fp_worker,
                           [(c, RADIUS, N_BITS) for c in chunks])

    flat_fps = [fp  for fps, _    in results for fp  in fps]
    flat_idx = [idx for _,   idxs in results for idx in idxs]
    print("  {:,} valid fingerprints".format(len(flat_fps)))
    return np.vstack(flat_fps), sample.loc[flat_idx].reset_index(drop=True)


# ------------------------------------------------------------------------------
# Plot 1: Score distributions
# ------------------------------------------------------------------------------

def plot_score_distributions(df: pd.DataFrame) -> None:
    print("\nPlotting score distributions ...")
    scores = df[SCORE_COL]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Docking Score Distribution  (n = {:,} compounds)".format(len(df)),
        y=1.01)

    # Full histogram
    ax = axes[0, 0]
    _ax_clean(ax)
    ax.hist(scores, bins=150, color=C["teal"], edgecolor="none", alpha=0.8)
    for pct, col, lbl in [(1,  C["red"],   "Top 1%"),
                           (5,  C["amber"], "Top 5%"),
                           (10, C["blue"],  "Top 10%")]:
        ax.axvline(scores.quantile(pct / 100), color=col,
                   lw=1.8, linestyle="--", label=lbl)
    ax.set_xlabel("Docking Score (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title("Full Distribution")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: "{:,.0f}".format(x)))

    # Log-scale
    ax = axes[0, 1]
    _ax_clean(ax)
    ax.hist(scores, bins=150, color=C["blue"], edgecolor="none", alpha=0.8,
            log=True)
    ax.set_xlabel("Docking Score (kcal/mol)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Full Distribution  (log y-axis)")

    # Decile boxplot
    ax = axes[1, 0]
    _ax_clean(ax, grid=False)
    df_tmp          = df[[SCORE_COL]].copy()
    df_tmp["decile"]= pd.qcut(scores, 10,
                               labels=["D{}".format(i) for i in range(1, 11)])
    decile_data = [df_tmp[df_tmp["decile"] == "D{}".format(i)][SCORE_COL].values
                   for i in range(1, 11)]
    bp = ax.boxplot(decile_data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", lw=2.5))
    grad = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 10))
    for patch, col in zip(bp["boxes"], grad):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
    for element in ["whiskers", "caps"]:
        for item in bp[element]:
            item.set_color(C["border"])
    for flier in bp["fliers"]:
        flier.set(marker=".", color=C["grey"], alpha=0.4, markersize=3)
    ax.set_xticklabels(
        ["D{}".format(i) for i in range(1, 11)])
    ax.set_xlabel("Score Decile  (D1 = best binders)")
    ax.set_ylabel("Docking Score (kcal/mol)")
    ax.set_title("Score by Decile")
    ax.grid(axis="y", color=C["lgrey"], linewidth=0.8)
    ax.set_axisbelow(True)

    # CDF
    ax = axes[1, 1]
    _ax_clean(ax)
    sorted_s = np.sort(scores)
    cdf      = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    ax.plot(sorted_s, cdf * 100, color=C["teal"], lw=2)
    for pct, col in [(1, C["red"]), (5, C["amber"]), (10, C["blue"])]:
        ax.axhline(pct, color=col, lw=1.2, linestyle="--",
                   label="{}%".format(pct))
    ax.set_xlabel("Docking Score (kcal/mol)")
    ax.set_ylabel("Cumulative % of compounds")
    ax.set_title("Cumulative Distribution")
    ax.legend(title="Threshold")

    plt.tight_layout()
    _save(fig, "01_score_distributions")


# ------------------------------------------------------------------------------
# Plot 2: PCA
# ------------------------------------------------------------------------------

def plot_pca(fp_matrix: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    print("\nRunning PCA ...")
    scaler = StandardScaler(with_mean=False)
    fps_sc = scaler.fit_transform(fp_matrix)
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(fps_sc)
    var    = pca.explained_variance_ratio_ * 100
    print("  PC1={:.1f}%  PC2={:.1f}%".format(var[0], var[1]))

    scores    = df[SCORE_COL].values
    vmin      = np.percentile(scores, 1)
    vmax      = np.percentile(scores, 99)
    top1_mask = scores <= np.percentile(scores, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "PCA of Morgan Fingerprints  (subsample n = {:,})".format(len(df)))

    # Coloured by score
    ax = axes[0]
    _ax_clean(ax, grid=False)
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=scores, cmap="RdYlGn_r",
                    vmin=vmin, vmax=vmax,
                    s=2, alpha=0.35, linewidths=0, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Docking Score (kcal/mol)")
    cbar.outline.set_edgecolor(C["border"])
    ax.set_xlabel("PC1  ({:.1f}% variance)".format(var[0]))
    ax.set_ylabel("PC2  ({:.1f}% variance)".format(var[1]))
    ax.set_title("Coloured by Docking Score")

    # Top 1% highlighted
    ax = axes[1]
    _ax_clean(ax, grid=False)
    ax.scatter(coords[~top1_mask, 0], coords[~top1_mask, 1],
               c=C["lgrey"], s=2, alpha=0.3, linewidths=0,
               label="Remaining compounds", rasterized=True)
    ax.scatter(coords[top1_mask, 0], coords[top1_mask, 1],
               c=C["red"], s=8, alpha=0.85, linewidths=0,
               label="Top 1%  (n={:,})".format(top1_mask.sum()),
               rasterized=True)
    ax.set_xlabel("PC1  ({:.1f}% variance)".format(var[0]))
    ax.set_ylabel("PC2  ({:.1f}% variance)".format(var[1]))
    ax.set_title("Top 1% Highlighted")
    ax.legend(markerscale=3)

    plt.tight_layout()
    _save(fig, "02_pca")
    return coords


# ------------------------------------------------------------------------------
# Plot 3: UMAP
# ------------------------------------------------------------------------------

def plot_umap(fp_matrix: np.ndarray, df: pd.DataFrame) -> None:
    if not UMAP_AVAILABLE:
        return
    print("\nRunning UMAP ...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                         metric="jaccard", random_state=42, low_memory=True)
    coords  = reducer.fit_transform(fp_matrix)

    scores    = df[SCORE_COL].values
    top1_mask = scores <= np.percentile(scores, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "UMAP of Morgan Fingerprints  (Jaccard metric,  n = {:,})".format(len(df)))

    ax = axes[0]
    _ax_clean(ax, grid=False)
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=scores, cmap="RdYlGn_r",
                    vmin=np.percentile(scores, 1),
                    vmax=np.percentile(scores, 99),
                    s=2, alpha=0.35, linewidths=0, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Docking Score (kcal/mol)")
    cbar.outline.set_edgecolor(C["border"])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Coloured by Docking Score")

    ax = axes[1]
    _ax_clean(ax, grid=False)
    ax.scatter(coords[~top1_mask, 0], coords[~top1_mask, 1],
               c=C["lgrey"], s=2, alpha=0.25, linewidths=0,
               label="Remaining", rasterized=True)
    ax.scatter(coords[top1_mask, 0], coords[top1_mask, 1],
               c=C["red"], s=8, alpha=0.85, linewidths=0,
               label="Top 1%  (n={:,})".format(top1_mask.sum()),
               rasterized=True)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Top 1% Highlighted")
    ax.legend(markerscale=3)

    plt.tight_layout()
    _save(fig, "03_umap")


# ------------------------------------------------------------------------------
# Plot 4: Property distributions
# ------------------------------------------------------------------------------

def plot_properties(props: pd.DataFrame) -> None:
    print("\nPlotting property distributions ...")

    prop_defs = [
        ("mw",       "Molecular Weight (Da)",   500,  "Ro5: MW ≤ 500"),
        ("logp",     "LogP",                     5,   "Ro5: LogP ≤ 5"),
        ("hbd",      "H-Bond Donors",            5,   "Ro5: HBD ≤ 5"),
        ("hba",      "H-Bond Acceptors",        10,   "Ro5: HBA ≤ 10"),
        ("tpsa",     "TPSA (Å²)",              140,   "Oral: TPSA ≤ 140"),
        ("rotbonds", "Rotatable Bonds",         10,   "Veber: RotBonds ≤ 10"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Molecular Property Distributions  (n = {:,})".format(len(props)))

    bar_colours = [C["teal"], C["blue"], C["amber"],
                   C["purple"], C["red"], C["grey"]]

    for ax, (col, label, limit, limit_label), bc in zip(
            axes.flat, prop_defs, bar_colours):
        _ax_clean(ax)
        vals = props[col].dropna()

        ax.hist(vals, bins=60, color=bc, edgecolor="none", alpha=0.75)
        ax.axvline(limit, color=C["red"], lw=1.8,
                   linestyle="--", label=limit_label)

        pct_ok = 100 * (vals <= limit).mean()
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title("{}\n{:.1f}% within limit".format(label, pct_ok))
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: "{:,.0f}".format(x)))

    plt.tight_layout()
    _save(fig, "04_property_distributions")


# ------------------------------------------------------------------------------
# Plot 5: Lipinski compliance
# ------------------------------------------------------------------------------

def plot_lipinski(props: pd.DataFrame) -> None:
    print("\nPlotting Lipinski compliance ...")

    labels  = ["0 violations\n(fully compliant)", "1 violation",
               "2 violations", "3 violations", "4 violations"]
    counts  = [(props["n_violations"] == i).sum() for i in range(5)]
    colours = [C["teal"], "#4CAF50", C["amber"], "#FF7043", C["red"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Lipinski Rule-of-5 Compliance")

    # Pie
    ax = axes[0]
    ax.set_facecolor("white")
    wedges, _, autos = ax.pie(
        counts, autopct="%1.1f%%", colors=colours,
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=1.5))
    for at in autos:
        at.set_fontsize(9)
        at.set_color(C["text"])
    ax.legend(wedges, labels,
              loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=9, framealpha=0.9,
              edgecolor=C["border"])
    ax.set_title("Violation Count Distribution\n(n={:,})".format(len(props)))

    # Per-rule bar
    ax = axes[1]
    _ax_clean(ax)
    rules = ["MW > 500", "LogP > 5", "HBD > 5", "HBA > 10"]
    pcts  = [100 * props["lip_{}".format(k)].mean()
             for k in ["mw", "logp", "hbd", "hba"]]
    bar_c = [C["teal"], C["blue"], C["amber"], C["purple"]]
    bars  = ax.barh(rules, pcts, color=bar_c,
                    edgecolor="white", height=0.5)
    for bar, pct in zip(bars, pcts):
        ax.text(pct + 0.4, bar.get_y() + bar.get_height() / 2,
                "{:.1f}%".format(pct), va="center",
                color=C["text"], fontsize=10, fontweight="bold")
    ax.set_xlabel("Compounds violating rule (%)")
    ax.set_title("Individual Rule Violations")
    ax.set_xlim(0, max(pcts) * 1.25)
    ax.grid(axis="x", color=C["lgrey"], linewidth=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save(fig, "05_lipinski_compliance")


# ------------------------------------------------------------------------------
# Plot 6: QED
# ------------------------------------------------------------------------------

def plot_qed(props: pd.DataFrame) -> None:
    print("\nPlotting QED ...")
    qed = props["qed"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "QED Drug-likeness  "
        "({:.1f}% of compounds with QED ≥ 0.5)".format(
            100 * (qed >= 0.5).mean()))

    ax = axes[0]
    _ax_clean(ax)
    ax.hist(qed, bins=80, color=C["teal"], edgecolor="none", alpha=0.8)
    ax.axvline(qed.median(), color=C["red"], lw=2,
               linestyle="--",
               label="Median  {:.3f}".format(qed.median()))
    ax.axvline(0.5, color=C["grey"], lw=1.5, linestyle=":",
               label="QED = 0.5  (guideline)")
    ax.set_xlabel("QED  (0 = least drug-like,  1 = most drug-like)")
    ax.set_ylabel("Count")
    ax.set_title("QED Distribution  (n={:,})".format(len(qed)))
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: "{:,.0f}".format(x)))

    # QED vs docking score
    ax = axes[1]
    _ax_clean(ax, grid=False)
    mask = props["qed"].notna() & props[SCORE_COL].notna()
    hb   = ax.hexbin(props.loc[mask, "qed"],
                     props.loc[mask, SCORE_COL],
                     gridsize=50, cmap="Blues",
                     mincnt=1, linewidths=0)
    cbar = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count")
    cbar.outline.set_edgecolor(C["border"])
    corr = props.loc[mask, ["qed", SCORE_COL]].corr().iloc[0, 1]
    ax.set_xlabel("QED")
    ax.set_ylabel("Docking Score (kcal/mol)")
    ax.set_title("QED vs Docking Score  (r = {:.3f})".format(corr))

    plt.tight_layout()
    _save(fig, "06_qed")


# ------------------------------------------------------------------------------
# Plot 7: Score-property correlations
# ------------------------------------------------------------------------------

def plot_score_correlations(props: pd.DataFrame) -> None:
    print("\nPlotting score-property correlations ...")

    prop_cols = ["mw",    "logp",  "hbd",  "hba",  "tpsa", "qed"]
    prop_lbls = ["MW",    "LogP",  "HBD",  "HBA",  "TPSA", "QED"]
    cmaps     = ["Blues", "Greens","Oranges","Purples","Reds","BuPu"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Docking Score vs Molecular Properties")

    for ax, col, lbl, cmap in zip(axes.flat, prop_cols, prop_lbls, cmaps):
        _ax_clean(ax, grid=False)
        mask = props[col].notna() & props[SCORE_COL].notna()
        hb   = ax.hexbin(props.loc[mask, col],
                         props.loc[mask, SCORE_COL],
                         gridsize=50, cmap=cmap,
                         mincnt=1, linewidths=0)
        cbar = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Count", fontsize=8)
        cbar.outline.set_edgecolor(C["border"])
        corr = props.loc[mask, [col, SCORE_COL]].corr().iloc[0, 1]
        ax.set_xlabel(lbl)
        ax.set_ylabel("Docking Score (kcal/mol)")
        ax.set_title("{}  vs  Score\nr = {:.3f}".format(lbl, corr))

    plt.tight_layout()
    _save(fig, "07_score_correlations")


# ------------------------------------------------------------------------------
# Plot 8: Scaffold diversity
# ------------------------------------------------------------------------------

def plot_scaffolds(df: pd.DataFrame) -> None:
    print("\nAnalysing scaffolds ({:,} sample) ...".format(N_FOR_SCAFFOLD))
    sample    = df.sample(min(N_FOR_SCAFFOLD, len(df)), random_state=42)
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

    sample         = sample.copy()
    sample["scaffold"] = scaffolds
    counts         = sample["scaffold"].value_counts()
    n_unique       = (counts.index != "INVALID").sum()
    singleton_pct  = 100 * (counts == 1).sum() / len(counts)
    print("  Unique scaffolds   : {:,}".format(n_unique))
    print("  Singleton scaffolds: {:.1f}%".format(singleton_pct))

    top15     = counts.head(15)
    other_cnt = counts.iloc[15:].sum()
    if other_cnt:
        top15["Other  ({:,} scaffolds)".format(len(counts) - 15)] = other_cnt

    colours = plt.cm.tab20(np.linspace(0, 1, len(top15)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Murcko Scaffold Diversity  "
        "({:,} unique scaffolds,  {:.1f}% singletons)".format(
            n_unique, singleton_pct))

    # Pie
    ax = axes[0]
    ax.set_facecolor("white")
    wedges, _, autos = ax.pie(
        top15.values, autopct="%1.1f%%", colors=colours,
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=1.2))
    for at in autos:
        at.set_fontsize(7.5)
    ax.legend(wedges,
              [s[:40] + "..." if len(s) > 40 else s
               for s in top15.index],
              loc="center left", bbox_to_anchor=(0.95, 0.5),
              fontsize=7, framealpha=0.9, edgecolor=C["border"])
    ax.set_title("Top Scaffold Distribution  (n={:,})".format(len(sample)))

    # Bar
    ax = axes[1]
    _ax_clean(ax)
    top20  = counts.head(20)
    colors = plt.cm.tab20(np.linspace(0, 1, len(top20)))
    ax.barh(range(len(top20)), top20.values,
            color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(
        [s[:42] + "..." if len(s) > 42 else s
         for s in top20.index], fontsize=7.5)
    ax.set_xlabel("Number of compounds")
    ax.set_title("Top 20 Most Common Scaffolds")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: "{:,.0f}".format(x)))

    plt.tight_layout()
    _save(fig, "08_scaffold_diversity")


# ------------------------------------------------------------------------------
# Plot 9: Internal diversity
# ------------------------------------------------------------------------------

def plot_internal_diversity(df: pd.DataFrame) -> Tuple[float, float]:
    print("\nCalculating diversity ({:,} sample) ...".format(N_FOR_DIVERSITY))
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
        sims = DataStructs.BulkTanimotoSimilarity(
            rdkit_fps[i], rdkit_fps[i + 1:])
        sim_vals.extend(sims)

    sim_arr   = np.array(sim_vals, dtype=np.float32)
    mean_sim  = float(sim_arr.mean())
    mean_dist = 1 - mean_sim
    print("  Mean pairwise Tanimoto similarity : {:.4f}".format(mean_sim))
    print("  Mean pairwise Tanimoto distance   : {:.4f}".format(mean_dist))

    # Nearest-neighbour
    nn_sims = []
    for i in range(n):
        all_sims = DataStructs.BulkTanimotoSimilarity(rdkit_fps[i], rdkit_fps)
        all_sims_arr = list(all_sims)
        all_sims_arr[i] = 0.0
        nn_sims.append(max(all_sims_arr))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Dataset Diversity  (subsample n = {:,},  "
        "mean Tanimoto distance = {:.3f})".format(n, mean_dist))

    ax = axes[0]
    _ax_clean(ax)
    ax.hist(sim_vals, bins=80, color=C["teal"],
            edgecolor="none", alpha=0.8)
    ax.axvline(mean_sim, color=C["red"], lw=2,
               linestyle="--",
               label="Mean  {:.3f}".format(mean_sim))
    ax.set_xlabel("Pairwise Tanimoto Similarity")
    ax.set_ylabel("Count")
    ax.set_title("All-pairs Similarity Distribution\n"
                 "Mean distance = {:.3f}".format(mean_dist))
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: "{:,.0f}".format(x)))

    ax = axes[1]
    _ax_clean(ax)
    ax.hist(nn_sims, bins=60, color=C["blue"],
            edgecolor="none", alpha=0.8)
    nn_mean = float(np.mean(nn_sims))
    ax.axvline(nn_mean, color=C["red"], lw=2,
               linestyle="--",
               label="Mean NN  {:.3f}".format(nn_mean))
    ax.set_xlabel("Nearest-Neighbour Tanimoto Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Nearest-Neighbour Similarity\n"
                 "(low values = isolated / unique compounds)")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: "{:,.0f}".format(x)))

    plt.tight_layout()
    _save(fig, "09_diversity")

    return mean_sim, mean_dist


# ------------------------------------------------------------------------------
# Structure grids  (PNG only -- molecule images are raster)
# ------------------------------------------------------------------------------

def plot_structure_grids(df: pd.DataFrame) -> None:
    print("\nDrawing structure grids ...")
    df_sorted = df.sort_values(SCORE_COL)

    for label, subset, fname in [
        ("top",    df_sorted.head(N_TOP_STRUCT),    "10_top_structures.png"),
        ("bottom", df_sorted.tail(N_BOTTOM_STRUCT), "11_bottom_structures.png"),
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

        out_path = os.path.join(OUTPUT_DIR, fname)
        img.save(out_path)
        size_kb = os.path.getsize(out_path) / 1024
        print("  PNG: {}  ({:.0f} KB)  [raster -- molecule images]".format(
            fname, size_kb))


# ------------------------------------------------------------------------------
# Summary text
# ------------------------------------------------------------------------------

def write_summary(df: pd.DataFrame,
                  props: pd.DataFrame,
                  mean_sim: float,
                  mean_dist: float) -> None:
    scores = df[SCORE_COL]
    lines  = [
        "Data Summary",
        "=" * 60,
        "",
        "Dataset",
        "-" * 40,
        "  Total compounds         : {:,}".format(len(df)),
        "",
        "Docking Scores",
        "-" * 40,
        "  Min  (best binder)      : {:.4f} kcal/mol".format(scores.min()),
        "  Max  (worst binder)     : {:.4f} kcal/mol".format(scores.max()),
        "  Mean                    : {:.4f} kcal/mol".format(scores.mean()),
        "  Std                     : {:.4f}".format(scores.std()),
        "  Median                  : {:.4f} kcal/mol".format(scores.median()),
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
        "Lipinski Rule-of-5",
        "-" * 40,
        "  Fully compliant (0 viol): {:.1f}%".format(
            100 * (props["n_violations"] == 0).mean()),
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
        "",
        "Output directory : {}".format(OUTPUT_DIR),
    ]

    out_path = os.path.join(OUTPUT_DIR, "00_summary.txt")
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

    df = load_data(TRAINING_CSV)

    plot_score_distributions(df)

    props = calculate_properties(df, n_sample=N_FOR_PROPS)
    plot_properties(props)
    plot_lipinski(props)
    plot_qed(props)
    plot_score_correlations(props)

    fp_matrix, fp_df = build_fp_matrix(df, N_FOR_PCA, label="PCA/UMAP")
    plot_pca(fp_matrix, fp_df)
    plot_umap(fp_matrix, fp_df)

    plot_scaffolds(df)
    mean_sim, mean_dist = plot_internal_diversity(df)
    plot_structure_grids(df)

    write_summary(df, props, mean_sim, mean_dist)

    print("\n" + "=" * 55)
    print("Analysis complete.  Outputs in '{}'".format(OUTPUT_DIR))
    print()
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        print("  {:50s}  {:>7.1f} KB".format(
            fname, os.path.getsize(fpath) / 1024))


if __name__ == "__main__":
    main()
