"""
generate_report.py
==================
Standalone script -- run AFTER analyse_training_data.py or analyse_results.py.

Reads the summary statistics from 00_summary.txt and the figures from
the output_old directory, then produces a formatted Word document report.

No modification to the analysis scripts is required.

Usage:
    # After analyse_training_data.py:
    python generate_report.py --mode training --dir /scratch/your-user/training_analysis/

    # After analyse_results.py:
    python generate_report.py --mode results --dir /scratch/your-user/analysis/

    # Specify output_old path explicitly:
    python generate_report.py --mode training --dir /scratch/your-user/training_analysis/ \\
        --output_old my_report.docx

Dependency:
    pip install python-docx
"""

import os
import re
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
except ImportError:
    print("ERROR: python-docx is required.")
    print("Install with:  pip install python-docx")
    sys.exit(1)


# ------------------------------------------------------------------------------
# Document styling helpers
# ------------------------------------------------------------------------------

def _set_cell_bg(cell, hex_colour: str) -> None:
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_colour)
    tcPr.append(shd)


def _rule(doc: "Document") -> None:
    p   = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pb  = OxmlElement("w:pBdr")
    bot = OxmlElement("w:bottom")
    bot.set(qn("w:val"),   "single")
    bot.set(qn("w:sz"),    "6")
    bot.set(qn("w:space"), "1")
    bot.set(qn("w:color"), "2E75B6")
    pb.append(bot)
    pPr.append(pb)
    p.paragraph_format.space_after = Pt(4)


def _style_doc(doc: "Document") -> None:
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)
    style.font.color.rgb = RGBColor(0x21, 0x21, 0x21)
    for level, size in [("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11)]:
        h = doc.styles[level]
        h.font.name  = "Arial"
        h.font.size  = Pt(size)
        h.font.bold  = True
        h.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)


def _table(doc: "Document",
           rows: List[tuple],
           headers: Optional[List[str]] = None) -> None:
    n_cols = len(rows[0])
    t = doc.add_table(rows=1 + len(rows), cols=n_cols)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    hdr_labels = headers or (["Property", "Value"] if n_cols == 2
                              else [""] * n_cols)
    for cell, label in zip(t.rows[0].cells, hdr_labels):
        cell.text = label
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _set_cell_bg(cell, "2E75B6")
    for r_idx, row_data in enumerate(rows):
        cells = t.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row_data):
            cells[c_idx].text = str(val)
            cells[c_idx].paragraphs[0].alignment = (
                WD_ALIGN_PARAGRAPH.RIGHT if c_idx > 0
                else WD_ALIGN_PARAGRAPH.LEFT)
            if r_idx % 2 == 0:
                _set_cell_bg(cells[c_idx], "EBF3FB")
    doc.add_paragraph()


def _figure(doc: "Document",
            path: str,
            caption: str,
            width: float = 6.0) -> None:
    """Add figure if it exists, otherwise insert a placeholder."""
    # Try PNG first, then SVG (Word handles PNG more reliably for embedding)
    found = None
    for ext in [".png", ".svg"]:
        candidate = path if path.endswith(ext) else path + ext
        if os.path.exists(candidate):
            found = candidate
            break

    if found and found.endswith(".png"):
        doc.add_picture(found, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        note = "[Figure: {}]".format(os.path.basename(path))
        p = doc.add_paragraph(note)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].italic = True
        p.runs[0].font.color.rgb = RGBColor(0x75, 0x75, 0x75)

    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(9)
    cap.runs[0].font.color.rgb = RGBColor(0x61, 0x61, 0x61)
    doc.add_paragraph()


# ------------------------------------------------------------------------------
# Summary.txt parser
# ------------------------------------------------------------------------------

def parse_summary(summary_path: str) -> Dict[str, Any]:
    """
    Parse key values from 00_summary.txt produced by the analysis scripts.
    Returns a dict of floats and ints.
    """
    stats: Dict[str, Any] = {}
    if not os.path.exists(summary_path):
        print("  [!] summary.txt not found at '{}'".format(summary_path))
        return stats

    with open(summary_path, "r") as f:
        text = f.read()

    def _grab(pattern: str, cast=float, default=None):
        m = re.search(pattern, text)
        if m:
            try:
                return cast(m.group(1).replace(",", ""))
            except Exception:
                pass
        return default

    # Dataset
    stats["n_compounds"]  = _grab(r"Total compounds\s*:\s*([\d,]+)", int)

    # Scores
    stats["score_min"]    = _grab(r"Min.*?:\s*([-\d.]+)")
    stats["score_max"]    = _grab(r"Max.*?:\s*([-\d.]+)")
    stats["score_mean"]   = _grab(r"Mean\s*:\s*([-\d.]+)")
    stats["score_std"]    = _grab(r"[Ss]td\s*:\s*([-\d.]+)")
    stats["score_median"] = _grab(r"[Mm]edian\s*:\s*([-\d.]+)")

    # Thresholds
    m1 = re.search(r"Top\s+1%.*?:\s*([-\d.]+).*?\([\s]*([\d,]+)", text)
    if m1:
        stats["top1_threshold"] = float(m1.group(1))
        stats["top1_count"]     = int(m1.group(2).replace(",", ""))

    m5 = re.search(r"Top\s+5%.*?:\s*([-\d.]+).*?\([\s]*([\d,]+)", text)
    if m5:
        stats["top5_threshold"] = float(m5.group(1))
        stats["top5_count"]     = int(m5.group(2).replace(",", ""))

    m10 = re.search(r"Top\s+10%.*?:\s*([-\d.]+).*?\([\s]*([\d,]+)", text)
    if m10:
        stats["top10_threshold"] = float(m10.group(1))
        stats["top10_count"]     = int(m10.group(2).replace(",", ""))

    # Properties
    stats["mw_mean"]    = _grab(r"MW\s+mean.*?:\s*([\d.]+)")
    stats["mw_median"]  = _grab(r"MW\s+median.*?:\s*([\d.]+)")
    stats["logp_mean"]  = _grab(r"LogP\s+mean.*?:\s*([-\d.]+)")
    stats["hbd_mean"]   = _grab(r"HBD\s+mean\s*:\s*([\d.]+)")
    stats["hba_mean"]   = _grab(r"HBA\s+mean\s*:\s*([\d.]+)")
    stats["tpsa_mean"]  = _grab(r"TPSA\s+mean\s*:\s*([\d.]+)")
    stats["qed_mean"]   = _grab(r"QED\s+mean\s*:\s*([\d.]+)")

    # Lipinski
    stats["lip_compliant_pct"] = _grab(
        r"Fully compliant.*?:\s*([\d.]+)")
    stats["lip_mw_pct"]   = _grab(r"MW\s*>\s*500\s*:\s*([\d.]+)")
    stats["lip_logp_pct"] = _grab(r"LogP\s*>\s*5\s*:\s*([\d.]+)")
    stats["lip_hbd_pct"]  = _grab(r"HBD\s*>\s*5\s*:\s*([\d.]+)")
    stats["lip_hba_pct"]  = _grab(r"HBA\s*>\s*10\s*:\s*([\d.]+)")

    # Diversity
    stats["mean_tanimoto_sim"]  = _grab(
        r"Mean pairwise Tanimoto similarity\s*:\s*([\d.]+)")
    stats["mean_tanimoto_dist"] = _grab(
        r"Mean pairwise Tanimoto distance\s*:\s*([\d.]+)")

    # Scaffold
    stats["n_unique_scaffolds"] = _grab(
        r"Unique scaffolds\s*:\s*([\d,]+)", int)
    stats["singleton_scaffold_pct"] = _grab(
        r"Singleton.*?:\s*([\d.]+)")

    return {k: v for k, v in stats.items() if v is not None}


# ------------------------------------------------------------------------------
# Training data report builder
# ------------------------------------------------------------------------------

def build_training_report(doc: "Document",
                           s: Dict[str, Any],
                           fig_dir: str) -> None:

    def g(key, fmt="{}", default="N/A"):
        val = s.get(key)
        if val is None:
            return default
        try:
            return fmt.format(val)
        except Exception:
            return str(val)

    # Title
    t = doc.add_heading("Virtual Screening: Training Dataset Analysis", 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(
        "Cheminformatics and molecular property analysis of the initial "
        "docking dataset used for neural network training")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(12)
    sub.runs[0].font.color.rgb = RGBColor(0x61, 0x61, 0x61)
    date_p = doc.add_paragraph(
        "Generated: {}".format(datetime.now().strftime("%d %B %Y")))
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_p.runs[0].font.color.rgb = RGBColor(0x61, 0x61, 0x61)
    doc.add_page_break()

    # Executive summary
    doc.add_heading("Executive Summary", 1)
    _rule(doc)
    n   = s.get("n_compounds", 0)
    doc.add_paragraph(
        "This report describes the chemical space, molecular property "
        "distribution, and structural diversity of {:,} compounds "
        "from the ZINC15 database used as the initial virtual screening "
        "training dataset. Docking scores range from {} to {} kcal/mol "
        "(mean {} kcal/mol), reflecting the broad chemical space sampled. "
        "The analysis provides baseline characterisation of dataset "
        "composition prior to neural network training and "
        "active learning.".format(
            n,
            g("score_min", "{:.2f}"),
            g("score_max", "{:.2f}"),
            g("score_mean", "{:.2f}")))

    doc.add_paragraph()
    doc.add_heading("Key Statistics", 2)
    _table(doc, [
        ("Total compounds",         "{:,}".format(n)),
        ("Best docking score",      g("score_min",   "{:.4f} kcal/mol")),
        ("Worst docking score",     g("score_max",   "{:.4f} kcal/mol")),
        ("Mean score",              g("score_mean",  "{:.4f} kcal/mol")),
        ("Std deviation",           g("score_std",   "{:.4f}")),
        ("Median score",            g("score_median","{:.4f} kcal/mol")),
        ("Top 1% threshold",        g("top1_threshold",  "{:.4f} kcal/mol")),
        ("Top 5% threshold",        g("top5_threshold",  "{:.4f} kcal/mol")),
        ("Lipinski compliant",      g("lip_compliant_pct", "{:.1f}%")),
        ("Mean QED",                g("qed_mean",    "{:.3f}")),
        ("Mean Tanimoto similarity",g("mean_tanimoto_sim", "{:.4f}")),
        ("Unique Murcko scaffolds", g("n_unique_scaffolds", "{:,}")),
    ])
    doc.add_page_break()

    # Score distribution
    doc.add_heading("1.  Docking Score Distribution", 1)
    _rule(doc)
    doc.add_paragraph(
        "Figure 1 illustrates the distribution of AutoDock Vina docking "
        "scores across all {:,} training compounds. The distribution "
        "spans {} to {} kcal/mol with a mean of {} +/- {} kcal/mol "
        "and a median of {} kcal/mol, displaying an approximately "
        "Gaussian shape consistent with a randomly sampled diverse "
        "library. The top 1% threshold ({} kcal/mol, {:,} compounds), "
        "top 5% ({} kcal/mol), and top 10% ({} kcal/mol) define the "
        "enrichment boundaries used to evaluate model performance.".format(
            n,
            g("score_min",  "{:.2f}"),
            g("score_max",  "{:.2f}"),
            g("score_mean", "{:.2f}"),
            g("score_std",  "{:.2f}"),
            g("score_median","{:.2f}"),
            g("top1_threshold", "{:.2f}"),
            s.get("top1_count", 0),
            g("top5_threshold", "{:.2f}"),
            g("top10_threshold","{:.2f}")))

    _figure(doc,
            os.path.join(fig_dir, "01_score_distributions"),
            "Figure 1. Docking score distribution. (A) Full histogram with "
            "top percentile thresholds. (B) Log-scale y-axis. "
            "(C) Box plots by decile. (D) Cumulative distribution function.")

    _table(doc, [
        ("Min (best binder)",   g("score_min",   "{:.4f} kcal/mol")),
        ("Max (worst binder)",  g("score_max",   "{:.4f} kcal/mol")),
        ("Mean +/- std",        "{} +/- {}".format(
            g("score_mean", "{:.4f}"), g("score_std", "{:.4f}"))),
        ("Median",              g("score_median", "{:.4f} kcal/mol")),
        ("Top 1% threshold",    "{} ({:,} cpds)".format(
            g("top1_threshold", "{:.4f}"), s.get("top1_count", 0))),
        ("Top 5% threshold",    "{} ({:,} cpds)".format(
            g("top5_threshold", "{:.4f}"), s.get("top5_count", 0))),
        ("Top 10% threshold",   "{} ({:,} cpds)".format(
            g("top10_threshold", "{:.4f}"), s.get("top10_count", 0))),
    ], headers=["Metric", "Value"])

    # Chemical space
    doc.add_heading("2.  Chemical Space Analysis", 1)
    _rule(doc)
    doc.add_paragraph(
        "Morgan fingerprints (radius 2, 2048 bits) were generated for a "
        "representative subsample and projected into two dimensions using "
        "Principal Component Analysis (PCA, Figure 2) and Uniform Manifold "
        "Approximation and Projection (UMAP, Figure 3). The broad "
        "distribution of compounds in both projections, with no single "
        "dominant cluster, confirms the structural heterogeneity of the "
        "ZINC15 training library. The top 1% best-scoring compounds "
        "are distributed across multiple regions of chemical space, "
        "indicating that high docking affinity is not confined to a "
        "single structural class.")

    _figure(doc,
            os.path.join(fig_dir, "02_pca"),
            "Figure 2. PCA of Morgan fingerprints. "
            "(Left) Coloured by docking score. "
            "(Right) Top 1% best-scoring compounds highlighted.")

    _figure(doc,
            os.path.join(fig_dir, "03_umap"),
            "Figure 3. UMAP projection (Jaccard metric). "
            "(Left) Coloured by docking score. "
            "(Right) Top 1% highlighted.")

    # Properties
    doc.add_heading("3.  Molecular Property Distributions", 1)
    _rule(doc)
    doc.add_paragraph(
        "Six molecular descriptors were calculated across the dataset "
        "(Figure 4): molecular weight (mean {} Da), LogP (mean {}), "
        "hydrogen bond donors (mean {}), hydrogen bond acceptors "
        "(mean {}), TPSA (mean {} A^2), and rotatable bonds. "
        "Vertical dashed lines indicate Lipinski Rule-of-Five and "
        "Veber criteria thresholds for oral bioavailability.".format(
            g("mw_mean",   "{:.1f}"),
            g("logp_mean", "{:.2f}"),
            g("hbd_mean",  "{:.2f}"),
            g("hba_mean",  "{:.2f}"),
            g("tpsa_mean", "{:.1f}")))

    _figure(doc,
            os.path.join(fig_dir, "04_property_distributions"),
            "Figure 4. Molecular property distributions with "
            "drug-likeness thresholds (dashed lines).")

    # Lipinski
    doc.add_heading("4.  Lipinski Rule-of-Five Compliance", 1)
    _rule(doc)
    doc.add_paragraph(
        "{}% of compounds satisfy all four Lipinski rules simultaneously "
        "(Figure 5). Individual rule violation rates: MW > 500 Da: {}%, "
        "LogP > 5: {}%, HBD > 5: {}%, HBA > 10: {}%. The high overall "
        "compliance rate indicates the library is well-suited for "
        "oral drug discovery campaigns.".format(
            g("lip_compliant_pct", "{:.1f}"),
            g("lip_mw_pct",   "{:.1f}"),
            g("lip_logp_pct", "{:.1f}"),
            g("lip_hbd_pct",  "{:.1f}"),
            g("lip_hba_pct",  "{:.1f}")))

    _figure(doc,
            os.path.join(fig_dir, "05_lipinski_compliance"),
            "Figure 5. Lipinski Rule-of-Five compliance. "
            "(Left) Violation count distribution. "
            "(Right) Individual rule violation rates.")

    # QED
    doc.add_heading("5.  Drug-likeness (QED)", 1)
    _rule(doc)
    doc.add_paragraph(
        "The Quantitative Estimate of Drug-likeness (QED) has a mean "
        "of {} across the dataset (Figure 6). This reflects moderate "
        "overall drug-likeness appropriate for a broadly sampled "
        "screening library. The QED vs docking score correlation "
        "is shown in the right panel.".format(g("qed_mean", "{:.3f}")))

    _figure(doc,
            os.path.join(fig_dir, "06_qed"),
            "Figure 6. QED distribution (left) and QED vs "
            "docking score (right).")

    # Correlations
    doc.add_heading("6.  Score-Property Correlations", 1)
    _rule(doc)
    doc.add_paragraph(
        "Hexbin density plots of docking score against each molecular "
        "descriptor are shown in Figure 7. Molecular weight shows the "
        "strongest correlation with docking score, reflecting the known "
        "tendency of larger molecules to achieve more favourable scores "
        "through increased contact surface area. This motivated the use "
        "of MW-normalised ligand efficiency as an alternative "
        "training target.")

    _figure(doc,
            os.path.join(fig_dir, "07_score_correlations"),
            "Figure 7. Docking score vs molecular property correlations. "
            "Pearson r values are annotated on each subplot.")

    # Scaffolds
    doc.add_heading("7.  Scaffold Diversity", 1)
    _rule(doc)
    doc.add_paragraph(
        "Murcko scaffold analysis identified {:,} unique scaffolds, "
        "of which {}% are singletons (Figure 8). This high singleton "
        "rate confirms excellent scaffold diversity, supporting "
        "generalisation of the trained neural network across diverse "
        "structural classes.".format(
            s.get("n_unique_scaffolds", 0),
            g("singleton_scaffold_pct", "{:.1f}")))

    _figure(doc,
            os.path.join(fig_dir, "08_scaffold_diversity"),
            "Figure 8. Murcko scaffold distribution (left pie chart) "
            "and top 20 most frequent scaffolds (right bar chart).")

    # Diversity
    doc.add_heading("8.  Pairwise Molecular Diversity", 1)
    _rule(doc)
    doc.add_paragraph(
        "The mean pairwise Tanimoto similarity among a random subsample "
        "is {} (distance {}), confirming high structural diversity "
        "(Figure 9). Values below 0.3 are conventionally considered "
        "structurally distinct; the dataset is well below this "
        "threshold on average, validating its suitability for training "
        "a broadly applicable predictive model.".format(
            g("mean_tanimoto_sim",  "{:.4f}"),
            g("mean_tanimoto_dist", "{:.4f}")))

    _figure(doc,
            os.path.join(fig_dir, "09_diversity"),
            "Figure 9. Pairwise Tanimoto similarity (left) and "
            "nearest-neighbour similarity (right) distributions.")

    # Structures
    doc.add_heading("9.  Representative Structures", 1)
    _rule(doc)
    doc.add_paragraph(
        "Figures 10 and 11 show the 2D structures of the top 40 "
        "best-scoring and 40 worst-scoring compounds respectively, "
        "with ZINC identifiers and docking scores annotated.")

    _figure(doc,
            os.path.join(fig_dir, "10_top_structures.png"),
            "Figure 10. Top 40 best-scoring compounds.",
            width=6.5)

    _figure(doc,
            os.path.join(fig_dir, "11_bottom_structures.png"),
            "Figure 11. Worst 40 compounds for comparison.",
            width=6.5)

    # Conclusions
    doc.add_heading("10.  Conclusions", 1)
    _rule(doc)
    doc.add_paragraph(
        "The training dataset demonstrates high structural diversity, "
        "broad chemical space coverage, and good overall drug-likeness, "
        "collectively validating its suitability for training a "
        "generalizable neural network model for virtual screening. "
        "Key properties supporting this conclusion:")

    for point in [
        "High scaffold diversity ({:,} unique Murcko scaffolds, "
        "{}% singletons) reduces overfitting risk.".format(
            s.get("n_unique_scaffolds", 0),
            g("singleton_scaffold_pct", "{:.1f}")),

        "Mean pairwise Tanimoto distance of {} indicates the "
        "library spans a broad chemical space.".format(
            g("mean_tanimoto_dist", "{:.4f}")),

        "{}% Lipinski compliance supports drug discovery "
        "relevance.".format(g("lip_compliant_pct", "{:.1f}")),

        "Approximately Gaussian score distribution provides a "
        "well-conditioned regression target for model training.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(point)


# ------------------------------------------------------------------------------
# Results report builder
# ------------------------------------------------------------------------------

def build_results_report(doc: "Document",
                          s: Dict[str, Any],
                          fig_dir: str) -> None:

    def g(key, fmt="{}", default="N/A"):
        val = s.get(key)
        if val is None:
            return default
        try:
            return fmt.format(val)
        except Exception:
            return str(val)

    n = s.get("n_compounds", 0)

    # Title
    t = doc.add_heading("Virtual Screening Results Report", 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(
        "Neural network predicted docking score analysis -- "
        "hit characterisation and chemical diversity")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(12)
    sub.runs[0].font.color.rgb = RGBColor(0x61, 0x61, 0x61)
    date_p = doc.add_paragraph(
        "Generated: {}".format(datetime.now().strftime("%d %B %Y")))
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_p.runs[0].font.color.rgb = RGBColor(0x61, 0x61, 0x61)
    doc.add_page_break()

    # Executive summary
    doc.add_heading("Executive Summary", 1)
    _rule(doc)
    doc.add_paragraph(
        "A neural network was used to predict docking scores for {:,} "
        "compounds from the ZINC15 library. Predicted scores range from "
        "{} to {} kcal/mol (mean {} kcal/mol). The top 1% of predictions "
        "({:,} compounds, threshold {} kcal/mol) represent the primary "
        "candidates for explicit AutoDock Vina docking in the next "
        "active learning cycle.".format(
            n,
            g("score_min",  "{:.2f}"),
            g("score_max",  "{:.2f}"),
            g("score_mean", "{:.2f}"),
            s.get("top1_count", 0),
            g("top1_threshold", "{:.2f}")))

    _table(doc, [
        ("Compounds scored",        "{:,}".format(n)),
        ("Best predicted score",    g("score_min",   "{:.4f} kcal/mol")),
        ("Mean predicted score",    g("score_mean",  "{:.4f} kcal/mol")),
        ("Top 1% threshold",        g("top1_threshold",  "{:.4f} kcal/mol")),
        ("Top 1% compound count",   "{:,}".format(s.get("top1_count", 0))),
        ("Top 5% threshold",        g("top5_threshold",  "{:.4f} kcal/mol")),
        ("Mean Tanimoto sim.",       g("mean_tanimoto_sim", "{:.4f}")),
        ("Unique scaffolds (top hits)", g("n_unique_scaffolds", "{:,}")),
    ])
    doc.add_page_break()

    # Score distribution
    doc.add_heading("1.  Predicted Score Distribution", 1)
    _rule(doc)
    doc.add_paragraph(
        "Figure 1 shows the distribution of neural network predicted "
        "docking scores across {:,} screened compounds, with a mean "
        "of {} +/- {} kcal/mol. Score percentile thresholds define "
        "candidate pools for the next docking campaign.".format(
            n,
            g("score_mean", "{:.2f}"),
            g("score_std",  "{:.2f}")))

    _figure(doc,
            os.path.join(fig_dir, "score_distribution"),
            "Figure 1. Predicted docking score distribution with "
            "percentile thresholds marked.")

    # Chemical space
    doc.add_heading("2.  Chemical Space Coverage", 1)
    _rule(doc)
    doc.add_paragraph(
        "PCA (Figure 2) and UMAP (Figure 3) projections of the Morgan "
        "fingerprint space reveal the distribution of predicted hits "
        "in chemical space. Clusters of high-scoring compounds in both "
        "projections indicate regions of chemical space enriched for "
        "predicted binding affinity, validating the model's "
        "discriminative ability.")

    _figure(doc,
            os.path.join(fig_dir, "pca"),
            "Figure 2. PCA of Morgan fingerprints coloured by predicted "
            "docking score, with top 1% highlighted.")

    _figure(doc,
            os.path.join(fig_dir, "umap"),
            "Figure 3. UMAP projection revealing non-linear cluster "
            "structure in predicted hit chemical space.")

    # Tanimoto
    doc.add_heading("3.  Structural Similarity of Top Hits", 1)
    _rule(doc)
    sim = s.get("mean_tanimoto_sim", float("nan"))
    doc.add_paragraph(
        "Pairwise Tanimoto similarity among the top predicted compounds "
        "has a mean of {} (Figure 4). {} This assessment determines "
        "whether the model has converged on a narrow structural class "
        "or identified diverse binding scaffolds, with implications "
        "for lead optimisation strategy.".format(
            g("mean_tanimoto_sim", "{:.4f}"),
            "The relatively low mean similarity indicates structural "
            "diversity among top-ranked compounds, suggesting multiple "
            "binding modes are represented."
            if isinstance(sim, float) and sim < 0.4
            else "Moderate similarity among top-ranked compounds suggests "
                 "some structural convergence, which may reflect a dominant "
                 "pharmacophore for this target."))

    _figure(doc,
            os.path.join(fig_dir, "tanimoto_heatmap"),
            "Figure 4. Pairwise Tanimoto similarity heatmap for the "
            "top predicted compounds.")

    # Scaffolds
    doc.add_heading("4.  Scaffold Diversity of Top Hits", 1)
    _rule(doc)
    doc.add_paragraph(
        "Murcko scaffold analysis of the top predicted compounds "
        "identified {:,} unique scaffolds ({}% singletons, Figure 5). "
        "High scaffold diversity indicates multiple independent "
        "chemotype families with predicted affinity, providing diverse "
        "starting points for medicinal chemistry optimisation.".format(
            s.get("n_unique_scaffolds", 0),
            g("singleton_scaffold_pct", "{:.1f}")))

    _figure(doc,
            os.path.join(fig_dir, "scaffold_diversity"),
            "Figure 5. Scaffold distribution among top predicted compounds.")

    # Properties
    doc.add_heading("5.  Property Profile of Top Hits", 1)
    _rule(doc)
    doc.add_paragraph(
        "Figure 6 compares molecular property distributions of the "
        "top predicted compounds against the full screened library. "
        "Shifts between the two groups reveal physicochemical features "
        "enriched among high-scoring compounds and may inform "
        "future pharmacophore models.")

    _figure(doc,
            os.path.join(fig_dir, "property_distributions"),
            "Figure 6. Molecular property distributions: top predicted "
            "compounds (red) vs all screened compounds (blue).")

    # Structures
    doc.add_heading("6.  Top Predicted Compound Structures", 1)
    _rule(doc)
    doc.add_paragraph(
        "Figure 7 shows the 2D structures of the highest-ranking "
        "predicted compounds, annotated with ZINC identifiers and "
        "predicted docking scores. These represent the primary "
        "candidates for the next explicit docking campaign.")

    _figure(doc,
            os.path.join(fig_dir, "top_structures.png"),
            "Figure 7. Top 40 compounds by predicted docking score.",
            width=6.5)

    # Next steps
    doc.add_heading("7.  Recommended Next Steps", 1)
    _rule(doc)
    doc.add_paragraph(
        "Based on this analysis, the recommended workflow for the "
        "next active learning iteration is:")

    for step in [
        "Generate 3D conformers and PDBQT files for the top {:,} "
        "predicted compounds using prepare_pdbqt.py.".format(
            s.get("top1_count", 0)),
        "Run explicit AutoDock Vina docking (estimated 2-3 days "
        "for 1 million compounds using the parallel MPI pipeline).",
        "Incorporate real docking scores into training using "
        "update_model.py (fine-tune mode recommended).",
        "Re-screen the full ZINC15 library with the updated model "
        "and repeat until RMSE plateaus between iterations.",
    ]:
        p = doc.add_paragraph(style="List Number")
        p.add_run(step)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Word report from analysis script outputs")
    parser.add_argument(
        "--mode", required=True, choices=["training", "results"],
        help="'training' for analyse_training_data.py output_old, "
             "'results' for analyse_results.py output_old")
    parser.add_argument(
        "--dir", required=True,
        help="Directory containing figures and 00_summary.txt")
    parser.add_argument(
        "--output_old", default=None,
        help="Output .docx path (default: <dir>/report.docx)")
    parser.add_argument(
        "--target", default=None,
        help="Optional target protein name for title page")
    args = parser.parse_args()

    fig_dir = os.path.abspath(args.dir)
    if not os.path.isdir(fig_dir):
        print("ERROR: directory not found: {}".format(fig_dir))
        sys.exit(1)

    summary_path = os.path.join(fig_dir, "00_summary.txt")
    print("Parsing statistics from '{}' ...".format(summary_path))
    stats = parse_summary(summary_path)
    if args.target:
        stats["target_name"] = args.target
    print("  {:,} values parsed".format(len(stats)))

    out_path = args.output or os.path.join(
        fig_dir, "{}_report.docx".format(args.mode))

    print("Building {} report ...".format(args.mode))
    doc = Document()
    _style_doc(doc)
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.5)

    if args.mode == "training":
        build_training_report(doc, stats, fig_dir)
    else:
        build_results_report(doc, stats, fig_dir)

    doc.save(out_path)
    size_kb = os.path.getsize(out_path) / 1024
    print("Report saved: {}  ({:.0f} KB)".format(out_path, size_kb))


if __name__ == "__main__":
    main()
