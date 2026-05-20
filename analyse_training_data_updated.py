"""
Top-Hit Chemical Space Analysis
=================================
Analyses the top-1M scoring compounds produced by 01_extract_top_hits.py,
generating a full set of publication-quality plots.

Output format:
  PNG  -- 300 DPI raster (default)
  SVG  -- vector, Word-insertable (add --save-svg)

Plots generated:
  01_score_distributions.png/svg
  02_pca.png/svg
  03_umap.png/svg
  04_property_distributions.png/svg
  05_lipinski_compliance.png/svg
  06_qed.png/svg
  07_score_correlations.png/svg
  08_scaffold_diversity.png/svg
  09_diversity.png/svg
  10_top_structures.png
  11_bottom_structures.png
  00_summary.txt

Parallelism:
  Property calculation  -- multiprocessing.Pool over SMILES chunks
  Fingerprint generation -- multiprocessing.Pool (existing)
  Tanimoto diversity    -- numpy batched row-chunks across Pool workers

Usage:
    # Minimal — path to CSV from 01_extract_top_hits.py
    python analyse_training_data_updated.py --input top_1M_hits.csv

    # Full control
    python analyse_training_data_updated.py \\
        --input    top_1M_hits.csv \\
        --outdir   ./analysis_round2/ \\
        --workers  24 \\
        --save-svg \\
        --no-umap

Dependencies:
    pip install rdkit pandas numpy matplotlib seaborn scikit-learn umap-learn
"""

import argparse
import math
import multiprocessing
import os
import warnings
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


# ── Defaults (all overrideable via CLI) ───────────────────────────────────────

_DEFAULT_INPUT   = "./Inference_1_Top_1M_Hits/top_1M_hits.csv"
_DEFAULT_OUTDIR  = None          # derived from --input directory when None

ZINC_COL    = "zinc_id"
SMILES_COL  = "smiles"
SCORE_COL   = "predicted_docking_score"

N_FOR_PCA       = 500_000
N_FOR_PROPS     = 500_000
N_FOR_DIVERSITY =  25_000   # pairwise O(n²); 25 K is statistically robust
N_FOR_SCAFFOLD  = 100_000
N_TOP_STRUCT    = 50
N_BOTTOM_STRUCT = 50

RADIUS     = 2
N_BITS     = 2048
CHUNK_SIZE = 50_000

SAVE_SVG   = False
PNG_DPI    = 300

# Set by parse_args() → main() before any function is called
N_WORKERS  = os.cpu_count() or 1
OUTPUT_DIR = _DEFAULT_OUTDIR


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chemical-space analysis of top-hit compounds from the "
                    "virtual-screening pipeline (01_extract_top_hits.py output)."
    )
    parser.add_argument(
        "--input", "-i",
        default=_DEFAULT_INPUT,
        help=f"Path to top-hits CSV (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--outdir", "-o",
        default=None,
        help="Output directory for plots and summary (default: same folder as --input)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int, default=os.cpu_count() or 1,
        help="Parallel worker processes (default: all logical CPUs)",
    )
    parser.add_argument(
        "--score-col",
        default=SCORE_COL,
        help=f"Score column name in input CSV (default: {SCORE_COL})",
    )
    parser.add_argument(
        "--n-pca",
        type=int, default=N_FOR_PCA,
        help=f"Compounds to subsample for PCA/UMAP (default: {N_FOR_PCA:,})",
    )
    parser.add_argument(
        "--n-props",
        type=int, default=N_FOR_PROPS,
        help=f"Compounds for property calculation (default: {N_FOR_PROPS:,})",
    )
    parser.add_argument(
        "--n-diversity",
        type=int, default=N_FOR_DIVERSITY,
        help=f"Compounds for pairwise diversity (default: {N_FOR_DIVERSITY:,})",
    )
    parser.add_argument(
        "--n-scaffold",
        type=int, default=N_FOR_SCAFFOLD,
        help=f"Compounds for scaffold analysis (default: {N_FOR_SCAFFOLD:,})",
    )
    parser.add_argument(
        "--save-svg",
        action="store_true",
        help="Also save SVG versions of each plot",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Skip the UMAP projection (saves time on very large subsamples)",
    )
    return parser.parse_args()


# ── Colour palette / style ────────────────────────────────────────────────────

C = {
    "teal":    "#00897B",
    "red":     "#E53935",
    "amber":   "#FB8C00",
    "blue":    "#1E88E5",
    "purple":  "#8E24AA",
    "grey":    "#757575",
    "lgrey":   "#EEEEEE",
    "border":  "#BDBDBD",
    "text":    "#212121",
    "subtext": "#616161",
}

plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     C["border"],
    "axes.labelcolor":    C["text"],
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.titlecolor":    C["text"],
    "axes.grid":          True,
    "axes.grid.which":    "major",
    "grid.color":         C["lgrey"],
    "grid.linewidth":     0.8,
    "xtick.color":        C["subtext"],
    "ytick.color":        C["subtext"],
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   C["border"],
    "legend.fontsize":    9,
    "figure.titlesize":   13,
    "figure.titleweight": "bold",
    "font.family":        "sans-serif",
    "text.color":         C["text"],
    "savefig.facecolor":  "white",
    "savefig.edgecolor":  "white",
})


# ── Shared helpers ────────────────────────────────────────────────────────────

def _ax_clean(ax, grid: bool = True) -> None:
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.patch.set_facecolor("white")

    if SAVE_SVG:
        p = os.path.join(OUTPUT_DIR, stem + ".svg")
        fig.savefig(p, format="svg", bbox_inches="tight", facecolor="white")
        print("  SVG: {}  ({:.0f} KB)".format(
            stem + ".svg", os.path.getsize(p) / 1024))

    p = os.path.join(OUTPUT_DIR, stem + ".png")
    fig.savefig(p, format="png", dpi=PNG_DPI,
                bbox_inches="tight", facecolor="white")
    print("  PNG: {}  ({:.0f} KB)".format(
        stem + ".png", os.path.getsize(p) / 1024))

    plt.close(fig)


def load_data(csv_path: str) -> pd.DataFrame:
    print("Loading '{}' ...".format(csv_path))
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=[SMILES_COL, SCORE_COL]).reset_index(drop=True)
    print("  {:,} compounds loaded".format(len(df)))
    print("  Score range : {:.4f}  to  {:.4f}".format(
        df[SCORE_COL].min(), df[SCORE_COL].max()))
    return df


# ── Property calculation (parallel) ──────────────────────────────────────────

def _calc_props_chunk(smiles_list: List[str]) -> pd.DataFrame:
    """Top-level worker: compute RDKit properties for a list of SMILES."""
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
                "_smi":     smi,   # retained temporarily for score alignment
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def calculate_properties(df: pd.DataFrame,
                          n_sample: int,
                          n_workers: int) -> pd.DataFrame:
    sample = df.sample(min(n_sample, len(df)), random_state=42)
    smiles = sample[SMILES_COL].tolist()
    scores = sample[SCORE_COL].tolist()

    chunks = [smiles[i: i + CHUNK_SIZE]
              for i in range(0, len(smiles), CHUNK_SIZE)]

    print("Calculating properties for {:,} compounds "
          "({} chunks, {} workers) ...".format(len(smiles), len(chunks), n_workers))

    with multiprocessing.Pool(processes=n_workers) as pool:
        frames = pool.map(_calc_props_chunk, chunks)

    props = pd.concat(frames, ignore_index=True)

    # Align scores: match valid SMILES in original order
    valid_smiles_set = set(props["_smi"].tolist())
    scores_aligned   = [s for smi, s in zip(smiles, scores)
                        if smi in valid_smiles_set]
    props[SCORE_COL] = scores_aligned[:len(props)]
    props = props.drop(columns=["_smi"])

    props["n_violations"] = (props["lip_mw"] + props["lip_logp"] +
                              props["lip_hbd"] + props["lip_hba"])
    print("  {:,} valid property rows".format(len(props)))
    return props


# ── Fingerprint generation (parallel) ────────────────────────────────────────

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
                    n_workers: int,
                    label: str = "") -> Tuple[np.ndarray, pd.DataFrame]:
    sample = df.sample(min(n_sample, len(df)), random_state=42).reset_index(drop=True)
    pairs  = list(zip(sample.index.tolist(), sample[SMILES_COL].tolist()))
    n_proc = min(n_workers, len(pairs))
    cs     = math.ceil(len(pairs) / n_proc)
    chunks = [pairs[i: i + cs] for i in range(0, len(pairs), cs)]

    print("Generating fingerprints{} ({:,} cpds, {} workers) ...".format(
        " for {}".format(label) if label else "", len(pairs), n_proc))

    with multiprocessing.Pool(processes=n_proc) as pool:
        results = pool.map(_fp_worker, [(c, RADIUS, N_BITS) for c in chunks])

    flat_fps = [fp  for fps, _    in results for fp  in fps]
    flat_idx = [idx for _,   idxs in results for idx in idxs]
    print("  {:,} valid fingerprints".format(len(flat_fps)))
    return np.vstack(flat_fps), sample.loc[flat_idx].reset_index(drop=True)


# ── Diversity (parallel numpy Tanimoto) ──────────────────────────────────────

def _tanimoto_chunk(args: Tuple) -> Tuple[List[float], List[float]]:
    """
    Worker: upper-triangle pairwise sims + NN sims for rows [r0 : r1].

    Tanimoto via numpy dot-product:
        T(a, b) = dot(a, b) / (|a| + |b| - dot(a, b))
    where a, b are binary (0/1) float32 vectors.
    """
    fps_f32, row_sums, r0, r1 = args
    n            = len(fps_f32)
    upper_sims   = []
    nn_sims      = []

    for i in range(r0, r1):
        fp_i = fps_f32[i]

        # Upper triangle: similarities to rows i+1 .. n-1
        if i + 1 < n:
            dots  = fps_f32[i + 1:] @ fp_i                       # (n-i-1,)
            union = row_sums[i] + row_sums[i + 1:] - dots
            sims  = np.where(union > 0, dots / union, 0.0)
            upper_sims.extend(sims.tolist())

        # Nearest-neighbour similarity (exclude self)
        dots_all  = fps_f32 @ fp_i                                # (n,)
        union_all = row_sums[i] + row_sums - dots_all
        sims_all  = np.where(union_all > 0, dots_all / union_all, 0.0)
        sims_all[i] = 0.0
        nn_sims.append(float(sims_all.max()))

    return upper_sims, nn_sims


def _compute_diversity_parallel(fps_f32: np.ndarray,
                                 n_workers: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel pairwise Tanimoto.  Returns (all_pairs_sims, nn_sims).
    Splits the row index into n_workers equal chunks; each worker
    handles its slice of the upper triangle.
    """
    n        = len(fps_f32)
    row_sums = fps_f32.sum(axis=1)                # (n,) — reused by all workers
    cs       = math.ceil(n / n_workers)
    tasks    = [
        (fps_f32, row_sums, i, min(i + cs, n))
        for i in range(0, n, cs)
    ]

    print("  Parallel Tanimoto: {} rows, {} workers, "
          "{} rows/worker ...".format(n, n_workers, cs))

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(_tanimoto_chunk, tasks)

    upper_sims = np.array(
        [s for upper, _ in results for s in upper], dtype=np.float32)
    nn_sims    = np.array(
        [s for _, nn in results for s in nn],    dtype=np.float32)

    return upper_sims, nn_sims


# ── Plot 1: Score distributions ───────────────────────────────────────────────

def plot_score_distributions(df: pd.DataFrame) -> None:
    print("\nPlotting score distributions ...")
    scores = df[SCORE_COL]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Predicted Docking Score Distribution  (n = {:,} compounds)".format(len(df)),
        y=1.01)

    ax = axes[0, 0]
    _ax_clean(ax)
    ax.hist(scores, bins=150, color=C["teal"], edgecolor="none", alpha=0.8)
    for pct, col, lbl in [(1,  C["red"],   "Top 1%"),
                           (5,  C["amber"], "Top 5%"),
                           (10, C["blue"],  "Top 10%")]:
        ax.axvline(scores.quantile(pct / 100), color=col,
                   lw=1.8, linestyle="--", label=lbl)
    ax.set_xlabel("Predicted Docking Score (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title("Full Distribution")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: "{:,.0f}".format(x)))

    ax = axes[0, 1]
    _ax_clean(ax)
    ax.hist(scores, bins=150, color=C["blue"], edgecolor="none", alpha=0.8, log=True)
    ax.set_xlabel("Predicted Docking Score (kcal/mol)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Full Distribution  (log y-axis)")

    ax = axes[1, 0]
    _ax_clean(ax, grid=False)
    df_tmp           = df[[SCORE_COL]].copy()
    df_tmp["decile"] = pd.qcut(scores, 10,
                                labels=["D{}".format(i) for i in range(1, 11)])
    decile_data = [df_tmp[df_tmp["decile"] == "D{}".format(i)][SCORE_COL].values
                   for i in range(1, 11)]
    bp   = ax.boxplot(decile_data, patch_artist=True, widths=0.6,
                      medianprops=dict(color="white", lw=2.5))
    grad = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 10))
    for patch, col in zip(bp["boxes"], grad):
        patch.set_facecolor(col); patch.set_alpha(0.85)
    for element in ["whiskers", "caps"]:
        for item in bp[element]:
            item.set_color(C["border"])
    for flier in bp["fliers"]:
        flier.set(marker=".", color=C["grey"], alpha=0.4, markersize=3)
    ax.set_xticklabels(["D{}".format(i) for i in range(1, 11)])
    ax.set_xlabel("Score Decile  (D1 = best binders)")
    ax.set_ylabel("Predicted Docking Score (kcal/mol)")
    ax.set_title("Score by Decile")
    ax.grid(axis="y", color=C["lgrey"], linewidth=0.8)
    ax.set_axisbelow(True)

    ax = axes[1, 1]
    _ax_clean(ax)
    sorted_s = np.sort(scores)
    cdf      = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    ax.plot(sorted_s, cdf * 100, color=C["teal"], lw=2)
    for pct, col in [(1, C["red"]), (5, C["amber"]), (10, C["blue"])]:
        ax.axhline(pct, color=col, lw=1.2, linestyle="--",
                   label="{}%".format(pct))
    ax.set_xlabel("Predicted Docking Score (kcal/mol)")
    ax.set_ylabel("Cumulative % of compounds")
    ax.set_title("Cumulative Distribution")
    ax.legend(title="Threshold")

    plt.tight_layout()
    _save(fig, "01_score_distributions")


# ── Plot 2: PCA ───────────────────────────────────────────────────────────────

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

    ax = axes[0]
    _ax_clean(ax, grid=False)
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=scores, cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
                    s=2, alpha=0.35, linewidths=0, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted Docking Score (kcal/mol)")
    cbar.outline.set_edgecolor(C["border"])
    ax.set_xlabel("PC1  ({:.1f}% variance)".format(var[0]))
    ax.set_ylabel("PC2  ({:.1f}% variance)".format(var[1]))
    ax.set_title("Coloured by Docking Score")

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


# ── Plot 3: UMAP ──────────────────────────────────────────────────────────────

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
    cbar.set_label("Predicted Docking Score (kcal/mol)")
    cbar.outline.set_edgecolor(C["border"])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
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
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("Top 1% Highlighted")
    ax.legend(markerscale=3)

    plt.tight_layout()
    _save(fig, "03_umap")


# ── Plot 4: Property distributions ───────────────────────────────────────────

def plot_properties(props: pd.DataFrame) -> None:
    print("\nPlotting property distributions ...")
    prop_defs = [
        ("mw",       "Molecular Weight (Da)",  500, "Ro5: MW ≤ 500"),
        ("logp",     "LogP",                    5,  "Ro5: LogP ≤ 5"),
        ("hbd",      "H-Bond Donors",           5,  "Ro5: HBD ≤ 5"),
        ("hba",      "H-Bond Acceptors",       10,  "Ro5: HBA ≤ 10"),
        ("tpsa",     "TPSA (Å²)",            140,  "Oral: TPSA ≤ 140"),
        ("rotbonds", "Rotatable Bonds",        10,  "Veber: RotBonds ≤ 10"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Molecular Property Distributions  (n = {:,})".format(len(props)))
    bar_colours = [C["teal"], C["blue"], C["amber"],
                   C["purple"], C["red"], C["grey"]]

    for ax, (col, label, limit, limit_label), bc in zip(
            axes.flat, prop_defs, bar_colours):
        _ax_clean(ax)
        vals    = props[col].dropna()
        ax.hist(vals, bins=60, color=bc, edgecolor="none", alpha=0.75)
        ax.axvline(limit, color=C["red"], lw=1.8, linestyle="--", label=limit_label)
        pct_ok  = 100 * (vals <= limit).mean()
        ax.set_xlabel(label); ax.set_ylabel("Count")
        ax.set_title("{}\n{:.1f}% within limit".format(label, pct_ok))
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: "{:,.0f}".format(x)))

    plt.tight_layout()
    _save(fig, "04_property_distributions")


# ── Plot 5: Lipinski ──────────────────────────────────────────────────────────

def plot_lipinski(props: pd.DataFrame) -> None:
    print("\nPlotting Lipinski compliance ...")
    labels  = ["0 violations\n(fully compliant)", "1 violation",
               "2 violations", "3 violations", "4 violations"]
    counts  = [(props["n_violations"] == i).sum() for i in range(5)]
    colours = [C["teal"], "#4CAF50", C["amber"], "#FF7043", C["red"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Lipinski Rule-of-5 Compliance")

    ax = axes[0]
    ax.set_facecolor("white")
    wedges, _, autos = ax.pie(
        counts, autopct="%1.1f%%", colors=colours, startangle=140,
        pctdistance=0.82, wedgeprops=dict(edgecolor="white", linewidth=1.5))
    for at in autos:
        at.set_fontsize(9); at.set_color(C["text"])
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=9, framealpha=0.9, edgecolor=C["border"])
    ax.set_title("Violation Count Distribution\n(n={:,})".format(len(props)))

    ax = axes[1]
    _ax_clean(ax)
    rules = ["MW > 500", "LogP > 5", "HBD > 5", "HBA > 10"]
    pcts  = [100 * props["lip_{}".format(k)].mean()
             for k in ["mw", "logp", "hbd", "hba"]]
    bar_c = [C["teal"], C["blue"], C["amber"], C["purple"]]
    bars  = ax.barh(rules, pcts, color=bar_c, edgecolor="white", height=0.5)
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


# ── Plot 6: QED ───────────────────────────────────────────────────────────────

def plot_qed(props: pd.DataFrame) -> None:
    print("\nPlotting QED ...")
    qed = props["qed"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "QED Drug-likeness  "
        "({:.1f}% of compounds with QED ≥ 0.5)".format(100 * (qed >= 0.5).mean()))

    ax = axes[0]
    _ax_clean(ax)
    ax.hist(qed, bins=80, color=C["teal"], edgecolor="none", alpha=0.8)
    ax.axvline(qed.median(), color=C["red"], lw=2, linestyle="--",
               label="Median  {:.3f}".format(qed.median()))
    ax.axvline(0.5, color=C["grey"], lw=1.5, linestyle=":",
               label="QED = 0.5  (guideline)")
    ax.set_xlabel("QED  (0 = least drug-like,  1 = most drug-like)")
    ax.set_ylabel("Count")
    ax.set_title("QED Distribution  (n={:,})".format(len(qed)))
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: "{:,.0f}".format(x)))

    ax = axes[1]
    _ax_clean(ax, grid=False)
    mask = props["qed"].notna() & props[SCORE_COL].notna()
    hb   = ax.hexbin(props.loc[mask, "qed"], props.loc[mask, SCORE_COL],
                     gridsize=50, cmap="Blues", mincnt=1, linewidths=0)
    cbar = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count"); cbar.outline.set_edgecolor(C["border"])
    corr = props.loc[mask, ["qed", SCORE_COL]].corr().iloc[0, 1]
    ax.set_xlabel("QED")
    ax.set_ylabel("Predicted Docking Score (kcal/mol)")
    ax.set_title("QED vs Docking Score  (r = {:.3f})".format(corr))

    plt.tight_layout()
    _save(fig, "06_qed")


# ── Plot 7: Score-property correlations ──────────────────────────────────────

def plot_score_correlations(props: pd.DataFrame) -> None:
    print("\nPlotting score-property correlations ...")
    prop_cols = ["mw",    "logp",  "hbd",  "hba",  "tpsa", "qed"]
    prop_lbls = ["MW",    "LogP",  "HBD",  "HBA",  "TPSA", "QED"]
    cmaps     = ["Blues", "Greens","Oranges","Purples","Reds","BuPu"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Predicted Docking Score vs Molecular Properties")

    for ax, col, lbl, cmap in zip(axes.flat, prop_cols, prop_lbls, cmaps):
        _ax_clean(ax, grid=False)
        mask = props[col].notna() & props[SCORE_COL].notna()
        hb   = ax.hexbin(props.loc[mask, col], props.loc[mask, SCORE_COL],
                         gridsize=50, cmap=cmap, mincnt=1, linewidths=0)
        cbar = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Count", fontsize=8)
        cbar.outline.set_edgecolor(C["border"])
        corr = props.loc[mask, [col, SCORE_COL]].corr().iloc[0, 1]
        ax.set_xlabel(lbl)
        ax.set_ylabel("Predicted Docking Score (kcal/mol)")
        ax.set_title("{}  vs  Score\nr = {:.3f}".format(lbl, corr))

    plt.tight_layout()
    _save(fig, "07_score_correlations")


# ── Plot 8: Scaffold diversity ────────────────────────────────────────────────

def plot_scaffolds(df: pd.DataFrame, n_scaffold: int) -> None:
    print("\nAnalysing scaffolds ({:,} sample) ...".format(n_scaffold))
    sample    = df.sample(min(n_scaffold, len(df)), random_state=42)
    scaffolds = []
    for smi in sample[SMILES_COL]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds.append("INVALID"); continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False)
            scaffolds.append(sc if sc else "NO_SCAFFOLD")
        except Exception:
            scaffolds.append("ERROR")

    sample             = sample.copy()
    sample["scaffold"] = scaffolds
    counts             = sample["scaffold"].value_counts()
    n_unique           = (counts.index != "INVALID").sum()
    singleton_pct      = 100 * (counts == 1).sum() / len(counts)
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

    ax = axes[0]
    ax.set_facecolor("white")
    wedges, _, autos = ax.pie(
        top15.values, autopct="%1.1f%%", colors=colours, startangle=140,
        pctdistance=0.82, wedgeprops=dict(edgecolor="white", linewidth=1.2))
    for at in autos:
        at.set_fontsize(7.5)
    ax.legend(wedges,
              [s[:40] + "..." if len(s) > 40 else s for s in top15.index],
              loc="center left", bbox_to_anchor=(0.95, 0.5),
              fontsize=7, framealpha=0.9, edgecolor=C["border"])
    ax.set_title("Top Scaffold Distribution  (n={:,})".format(len(sample)))

    ax = axes[1]
    _ax_clean(ax)
    top20  = counts.head(20)
    colors = plt.cm.tab20(np.linspace(0, 1, len(top20)))
    ax.barh(range(len(top20)), top20.values, color=colors,
            edgecolor="white", height=0.7)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(
        [s[:42] + "..." if len(s) > 42 else s for s in top20.index],
        fontsize=7.5)
    ax.set_xlabel("Number of compounds")
    ax.set_title("Top 20 Most Common Scaffolds")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: "{:,.0f}".format(x)))

    plt.tight_layout()
    _save(fig, "08_scaffold_diversity")


# ── Plot 9: Internal diversity ────────────────────────────────────────────────

def plot_internal_diversity(df: pd.DataFrame,
                             n_diversity: int,
                             n_workers: int) -> Tuple[float, float]:
    print("\nCalculating diversity ({:,} sample) ...".format(n_diversity))
    sample   = df.sample(min(n_diversity, len(df)), random_state=42)
    gen      = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps_list = []
    for smi in sample[SMILES_COL]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps_list.append(
                gen.GetFingerprintAsNumPy(mol).astype(np.float32))

    fps_f32  = np.vstack(fps_list)
    n        = len(fps_f32)
    print("  {:,} valid fingerprints — launching parallel Tanimoto ...".format(n))

    sim_arr, nn_arr = _compute_diversity_parallel(fps_f32, n_workers)

    mean_sim  = float(sim_arr.mean())
    mean_dist = 1.0 - mean_sim
    print("  Mean pairwise Tanimoto similarity : {:.4f}".format(mean_sim))
    print("  Mean pairwise Tanimoto distance   : {:.4f}".format(mean_dist))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Dataset Diversity  (subsample n = {:,},  "
        "mean Tanimoto distance = {:.3f})".format(n, mean_dist))

    ax = axes[0]
    _ax_clean(ax)
    ax.hist(sim_arr, bins=80, color=C["teal"], edgecolor="none", alpha=0.8)
    ax.axvline(mean_sim, color=C["red"], lw=2, linestyle="--",
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
    nn_mean = float(nn_arr.mean())
    ax.hist(nn_arr, bins=60, color=C["blue"], edgecolor="none", alpha=0.8)
    ax.axvline(nn_mean, color=C["red"], lw=2, linestyle="--",
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


# ── Structure grids ───────────────────────────────────────────────────────────

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

        img      = Draw.MolsToGridImage(
            mols, molsPerRow=8, subImgSize=(300, 250),
            legends=legends, returnPNG=False)
        out_path = os.path.join(OUTPUT_DIR, fname)
        img.save(out_path)
        print("  PNG: {}  ({:.0f} KB)  [molecule grid]".format(
            fname, os.path.getsize(out_path) / 1024))


# ── Summary text ──────────────────────────────────────────────────────────────

def write_summary(df: pd.DataFrame,
                  props: pd.DataFrame,
                  mean_sim: float,
                  mean_dist: float,
                  input_path: str) -> None:
    scores = df[SCORE_COL]
    lines  = [
        "Top-Hit Data Summary",
        "=" * 60,
        "",
        "Input file",
        "-" * 40,
        "  {}".format(os.path.abspath(input_path)),
        "",
        "Dataset",
        "-" * 40,
        "  Total compounds         : {:,}".format(len(df)),
        "",
        "Predicted Docking Scores",
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
        "  TPSA mean               : {:.1f} Å²".format(props.tpsa.mean()),
        "  QED  mean               : {:.3f}".format(props.qed.mean()),
        "",
        "Lipinski Rule-of-5",
        "-" * 40,
        "  Fully compliant (0 viol): {:.1f}%".format(
            100 * (props["n_violations"] == 0).mean()),
        "  MW > 500                : {:.1f}%".format(100 * props.lip_mw.mean()),
        "  LogP > 5                : {:.1f}%".format(100 * props.lip_logp.mean()),
        "  HBD > 5                 : {:.1f}%".format(100 * props.lip_hbd.mean()),
        "  HBA > 10                : {:.1f}%".format(100 * props.lip_hba.mean()),
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    multiprocessing.freeze_support()

    args = parse_args()

    # ── Wire CLI args into module-level config ────────────────────────────────
    global OUTPUT_DIR, N_WORKERS, SAVE_SVG, SCORE_COL, UMAP_AVAILABLE

    N_WORKERS  = args.workers
    SAVE_SVG   = args.save_svg
    SCORE_COL  = args.score_col

    if args.no_umap:
        UMAP_AVAILABLE = False

    OUTPUT_DIR = (
        args.outdir
        if args.outdir
        else os.path.dirname(os.path.abspath(args.input))
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Input      : {}".format(args.input))
    print("Output dir : {}".format(OUTPUT_DIR))
    print("Workers    : {}".format(N_WORKERS))
    print("Score col  : {}".format(SCORE_COL))
    print("=" * 60 + "\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_data(args.input)

    # ── Score distributions ───────────────────────────────────────────────────
    plot_score_distributions(df)

    # ── Molecular properties (parallel) ──────────────────────────────────────
    props = calculate_properties(df, n_sample=args.n_props, n_workers=N_WORKERS)
    plot_properties(props)
    plot_lipinski(props)
    plot_qed(props)
    plot_score_correlations(props)

    # ── PCA + UMAP (fingerprint matrix shared) ────────────────────────────────
    fp_matrix, fp_df = build_fp_matrix(
        df, args.n_pca, n_workers=N_WORKERS, label="PCA/UMAP")
    plot_pca(fp_matrix, fp_df)
    plot_umap(fp_matrix, fp_df)

    # ── Scaffolds ─────────────────────────────────────────────────────────────
    plot_scaffolds(df, n_scaffold=args.n_scaffold)

    # ── Diversity (parallel numpy Tanimoto) ───────────────────────────────────
    mean_sim, mean_dist = plot_internal_diversity(
        df, n_diversity=args.n_diversity, n_workers=N_WORKERS)

    # ── Structure grids ───────────────────────────────────────────────────────
    plot_structure_grids(df)

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary(df, props, mean_sim, mean_dist, input_path=args.input)

    print("\n" + "=" * 60)
    print("Analysis complete.  Outputs in '{}'".format(OUTPUT_DIR))
    print()
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            print("  {:50s}  {:>7.1f} KB".format(
                fname, os.path.getsize(fpath) / 1024))


if __name__ == "__main__":
    main()
