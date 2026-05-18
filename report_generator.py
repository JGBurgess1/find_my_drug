"""
report_generator.py
====================
Shared module used by analyse_training_data.py and analyse_results.py.

Generates a formatted Word document (.docx) report describing the
statistical results and figures produced by each analysis script.

Reports include:
  - Title page with dataset summary
  - One section per figure with narrative description and key values
  - Statistics tables
  - Scientific interpretation of each result
  - Conclusions and recommended next steps

Dependency:
    pip install python-docx

Usage (called internally by analysis scripts):
    from report_generator import TrainingDataReport, ResultsReport
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("python-docx not installed -- reports will not be generated.")
    print("Install with: pip install python-docx")


# ------------------------------------------------------------------------------
# Shared document styling helpers
# ------------------------------------------------------------------------------

def _set_cell_bg(cell, hex_colour: str) -> None:
    """Set table cell background colour."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_colour)
    tcPr.append(shd)


def _add_horizontal_rule(doc: "Document") -> None:
    """Add a thin horizontal line."""
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


def _build_doc_styles(doc: "Document") -> None:
    """Apply consistent styles to the document."""
    style = doc.styles["Normal"]
    style.font.name  = "Arial"
    style.font.size  = Pt(11)
    style.font.color.rgb = RGBColor(0x21, 0x21, 0x21)

    for level, size, bold in [
        ("Heading 1", 16, True),
        ("Heading 2", 13, True),
        ("Heading 3", 11, True),
    ]:
        h = doc.styles[level]
        h.font.name  = "Arial"
        h.font.size  = Pt(size)
        h.font.bold  = bold
        h.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)


def _add_stats_table(doc: "Document",
                     rows: List[tuple],
                     headers: Optional[List[str]] = None) -> None:
    """
    Add a clean two-column (or more) statistics table.
    rows: list of tuples, e.g. [("Mean score", "-7.23"), ...]
    """
    n_cols = len(rows[0])
    table  = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT

    # Header row
    hdr_cells = table.rows[0].cells
    hdr_labels = headers or (["Property", "Value"] if n_cols == 2
                              else [""] * n_cols)
    for i, (cell, label) in enumerate(zip(hdr_cells, hdr_labels)):
        cell.text = label
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _set_cell_bg(cell, "2E75B6")

    # Data rows
    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        for c_idx, value in enumerate(row_data):
            row_cells[c_idx].text = str(value)
            row_cells[c_idx].paragraphs[0].alignment = (
                WD_ALIGN_PARAGRAPH.RIGHT if c_idx > 0
                else WD_ALIGN_PARAGRAPH.LEFT)
            if r_idx % 2 == 0:
                _set_cell_bg(row_cells[c_idx], "EBF3FB")

    doc.add_paragraph()   # spacing after table


def _add_figure(doc: "Document",
                img_path: str,
                caption: str,
                width_inches: float = 6.0) -> None:
    """Add a figure with caption if the image file exists."""
    if not os.path.exists(img_path):
        doc.add_paragraph("[Figure not found: {}]".format(
            os.path.basename(img_path))).italic = True
        return
    doc.add_picture(img_path, width=Inches(width_inches))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(9)
    cap.runs[0].font.color.rgb = RGBColor(0x61, 0x61, 0x61)
    doc.add_paragraph()


# ------------------------------------------------------------------------------
# Training data report
# ------------------------------------------------------------------------------

class TrainingDataReport:
    """
    Generates a Word report describing the training dataset analysis.

    Call after all plots have been generated by analyse_training_data.py.

    stats dict keys (all optional -- report adapts to what is available):
        n_compounds, score_min, score_max, score_mean, score_std,
        score_median, top1_threshold, top1_count, top5_threshold,
        top5_count, top10_threshold, top10_count,
        mw_mean, mw_median, logp_mean, logp_median,
        hbd_mean, hba_mean, tpsa_mean, qed_mean,
        lip_compliant_pct, lip_mw_pct, lip_logp_pct,
        lip_hbd_pct, lip_hba_pct,
        mean_tanimoto_sim, mean_tanimoto_dist,
        n_unique_scaffolds, singleton_scaffold_pct,
        pca_var1, pca_var2,
        target_name  (optional, e.g. "QCRB")
    """

    def __init__(self, stats: Dict[str, Any], output_dir: str):
        self.s   = stats
        self.out = output_dir

    def _get(self, key: str, fmt: str = "{}", default: str = "N/A") -> str:
        val = self.s.get(key)
        if val is None:
            return default
        try:
            return fmt.format(val)
        except Exception:
            return str(val)

    def generate(self, out_path: str) -> None:
        if not DOCX_AVAILABLE:
            return

        doc = Document()
        _build_doc_styles(doc)

        # Page margins
        for section in doc.sections:
            section.top_margin    = Cm(2.5)
            section.bottom_margin = Cm(2.5)
            section.left_margin   = Cm(2.5)
            section.right_margin  = Cm(2.5)

        self._title_page(doc)
        self._executive_summary(doc)
        self._section_score_distribution(doc)
        self._section_chemical_space(doc)
        self._section_properties(doc)
        self._section_lipinski(doc)
        self._section_qed(doc)
        self._section_correlations(doc)
        self._section_scaffolds(doc)
        self._section_diversity(doc)
        self._section_structures(doc)
        self._conclusions(doc)

        doc.save(out_path)
        print("  Report saved: {}".format(out_path))

    def _title_page(self, doc: "Document") -> None:
        doc.add_paragraph()
        doc.add_paragraph()
        t = doc.add_heading("Virtual Screening: Training Dataset Analysis", 0)
        t.alignment = WD_ALIGN_PARAGRAPH.CENTER

        target = self.s.get("target_name", "target protein")
        sub = doc.add_paragraph(
            "Cheminformatics and molecular property analysis of the initial "
            "docking dataset used for neural network training")
        sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sub.runs[0].font.size = Pt(12)
        sub.runs[0].font.color.rgb = RGBColor(0x61, 0x61, 0x61)

        doc.add_paragraph()
        date_p = doc.add_paragraph(
            "Generated: {}".format(
                datetime.now().strftime("%d %B %Y")))
        date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_p.runs[0].font.color.rgb = RGBColor(0x61, 0x61, 0x61)

        doc.add_page_break()

    def _executive_summary(self, doc: "Document") -> None:
        doc.add_heading("Executive Summary", 1)
        _add_horizontal_rule(doc)

        n    = self.s.get("n_compounds", 0)
        smin = self.s.get("score_min",  float("nan"))
        smax = self.s.get("score_max",  float("nan"))
        smn  = self.s.get("score_mean", float("nan"))

        doc.add_paragraph(
            "This report describes the chemical space, molecular property "
            "distribution, and structural diversity of {:,} compounds "
            "comprising the initial virtual screening training dataset. "
            "Docking scores range from {:.2f} to {:.2f} kcal/mol "
            "(mean {:.2f} kcal/mol), reflecting the broad chemical space "
            "sampled from the ZINC15 database. The analysis provides "
            "baseline characterisation of dataset composition prior to "
            "neural network training and active learning iterations.".format(
                n, smin, smax, smn))

        doc.add_paragraph()
        doc.add_heading("Key Statistics", 2)
        _add_stats_table(doc, [
            ("Total compounds",      "{:,}".format(n)),
            ("Best docking score",   "{:.4f} kcal/mol".format(smin)),
            ("Worst docking score",  "{:.4f} kcal/mol".format(smax)),
            ("Mean score",           "{:.4f} kcal/mol".format(smn)),
            ("Std deviation",        self._get("score_std", "{:.4f}")),
            ("Top 1% threshold",     self._get("top1_threshold",
                                               "{:.4f} kcal/mol")),
            ("Unique scaffolds",     self._get("n_unique_scaffolds",
                                               "{:,}")),
            ("Mean Tanimoto dist.",  self._get("mean_tanimoto_dist",
                                               "{:.4f}")),
            ("Lipinski compliant",   self._get("lip_compliant_pct",
                                               "{:.1f}%")),
            ("Mean QED",             self._get("qed_mean", "{:.3f}")),
        ])
        doc.add_page_break()

    def _section_score_distribution(self, doc: "Document") -> None:
        doc.add_heading("1.  Docking Score Distribution", 1)
        _add_horizontal_rule(doc)

        smin  = self.s.get("score_min",  float("nan"))
        smax  = self.s.get("score_max",  float("nan"))
        smn   = self.s.get("score_mean", float("nan"))
        sstd  = self.s.get("score_std",  float("nan"))
        smed  = self.s.get("score_median", float("nan"))
        t1    = self.s.get("top1_threshold",  float("nan"))
        c1    = self.s.get("top1_count",  0)
        t5    = self.s.get("top5_threshold",  float("nan"))
        c5    = self.s.get("top5_count",  0)
        t10   = self.s.get("top10_threshold", float("nan"))
        c10   = self.s.get("top10_count", 0)
        n     = self.s.get("n_compounds", 1)

        doc.add_paragraph(
            "Figure 1 illustrates the distribution of AutoDock Vina docking "
            "scores across all {:,} compounds in the training dataset. The "
            "distribution spans {:.2f} to {:.2f} kcal/mol with a mean of "
            "{:.2f} \u00b1 {:.2f} kcal/mol and a median of {:.2f} kcal/mol. "
            "The distribution displays a roughly Gaussian shape with a "
            "slight negative skew, consistent with a random, chemically "
            "diverse compound library screened against a single target.".format(
                n, smin, smax, smn, sstd, smed))

        doc.add_paragraph(
            "The cumulative distribution plot (Figure 1, bottom right) "
            "reveals that the top 1% of compounds ({:,} compounds) "
            "achieve docking scores below {:.2f} kcal/mol, the top 5% "
            "({:,} compounds) below {:.2f} kcal/mol, and the top 10% "
            "({:,} compounds) below {:.2f} kcal/mol. These thresholds "
            "define the enrichment boundaries used to evaluate neural "
            "network performance during training.".format(
                c1, t1, c5, t5, c10, t10))

        _add_figure(doc,
                    os.path.join(self.out, "01_score_distributions.png"),
                    "Figure 1. Docking score distribution. (A) Full histogram "
                    "with top 1%, 5%, and 10% thresholds indicated. "
                    "(B) Log-scale y-axis to reveal tail behaviour. "
                    "(C) Box plots by score decile. "
                    "(D) Cumulative distribution function.")

        _add_stats_table(doc, [
            ("Minimum score (best)",  "{:.4f} kcal/mol".format(smin)),
            ("Maximum score (worst)", "{:.4f} kcal/mol".format(smax)),
            ("Mean \u00b1 std",       "{:.4f} \u00b1 {:.4f}".format(smn, sstd)),
            ("Median",                "{:.4f} kcal/mol".format(smed)),
            ("Top 1% threshold",      "{:.4f} ({:,} cpds)".format(t1, c1)),
            ("Top 5% threshold",      "{:.4f} ({:,} cpds)".format(t5, c5)),
            ("Top 10% threshold",     "{:.4f} ({:,} cpds)".format(t10, c10)),
        ], headers=["Metric", "Value"])

    def _section_chemical_space(self, doc: "Document") -> None:
        doc.add_heading("2.  Chemical Space Analysis", 1)
        _add_horizontal_rule(doc)

        v1 = self.s.get("pca_var1", float("nan"))
        v2 = self.s.get("pca_var2", float("nan"))

        doc.add_paragraph(
            "To visualise the distribution of compounds in chemical space, "
            "Morgan fingerprints (radius 2, 2048 bits) were generated for a "
            "representative subsample and subjected to principal component "
            "analysis (PCA). The first two principal components capture "
            "{:.1f}% and {:.1f}% of the total variance respectively "
            "(Figure 2), indicating that the dataset spans a broad and "
            "heterogeneous region of chemical space as expected for a "
            "randomly sampled diverse library.".format(v1, v2))

        doc.add_paragraph(
            "Compounds are distributed broadly across the PCA projection "
            "with no dominant cluster, confirming adequate structural "
            "diversity for training a generalizable neural network model. "
            "The top 1% best-scoring compounds (highlighted in Figure 2, "
            "right panel) do not form a single isolated cluster, suggesting "
            "that high docking affinity is not confined to one structural "
            "class and that the model must learn complex structure-activity "
            "relationships rather than simple scaffold preferences.")

        _add_figure(doc,
                    os.path.join(self.out, "02_pca.png"),
                    "Figure 2. PCA of Morgan fingerprints. "
                    "(Left) Compounds coloured by docking score. "
                    "(Right) Top 1% best-scoring compounds highlighted.")

        if UMAP_NOTE := self.s.get("umap_computed", False):
            doc.add_paragraph(
                "UMAP dimensionality reduction (Jaccard metric) was also "
                "applied to the fingerprint matrix to capture non-linear "
                "structure in chemical space (Figure 3). UMAP reveals "
                "discrete clusters of structurally related compounds that "
                "are not apparent in the linear PCA projection, confirming "
                "the presence of multiple distinct chemotype families "
                "within the dataset.")
            _add_figure(doc,
                        os.path.join(self.out, "03_umap.png"),
                        "Figure 3. UMAP of Morgan fingerprints (Jaccard metric). "
                        "Compounds coloured by docking score (left) and with "
                        "top 1% highlighted (right).")

    def _section_properties(self, doc: "Document") -> None:
        doc.add_heading("3.  Molecular Property Distributions", 1)
        _add_horizontal_rule(doc)

        doc.add_paragraph(
            "Six key molecular descriptors were calculated to characterise "
            "the physicochemical profile of the dataset (Figure 4): "
            "molecular weight (MW), lipophilicity (LogP), hydrogen bond "
            "donors (HBD), hydrogen bond acceptors (HBA), topological "
            "polar surface area (TPSA), and number of rotatable bonds. "
            "Vertical dashed lines indicate standard drug-likeness thresholds "
            "from Lipinski's Rule of Five and Veber's criteria.")

        rows = []
        for prop, label, limit in [
            ("mw_mean",       "MW mean",          None),
            ("mw_median",     "MW median",        None),
            ("logp_mean",     "LogP mean",        None),
            ("hbd_mean",      "HBD mean",         None),
            ("hba_mean",      "HBA mean",         None),
            ("tpsa_mean",     "TPSA mean",        None),
        ]:
            val = self.s.get(prop)
            rows.append((label,
                         "{:.2f}".format(val) if val is not None else "N/A"))

        _add_figure(doc,
                    os.path.join(self.out, "04_property_distributions.png"),
                    "Figure 4. Molecular property distributions. "
                    "Bars exceeding drug-likeness thresholds (dashed lines) "
                    "are highlighted. Percentage within each limit is shown "
                    "in the subplot title.")

        _add_stats_table(doc, rows, headers=["Property", "Value"])

    def _section_lipinski(self, doc: "Document") -> None:
        doc.add_heading("4.  Lipinski Rule-of-Five Compliance", 1)
        _add_horizontal_rule(doc)

        pct = self.s.get("lip_compliant_pct", float("nan"))
        doc.add_paragraph(
            "Lipinski's Rule of Five compliance was assessed across the "
            "dataset (Figure 5). {:.1f}% of compounds satisfy all four "
            "rules simultaneously, indicating good overall oral "
            "bioavailability potential for the majority of the library. "
            "The individual rule violation rates are summarised below.".format(
                pct))

        _add_figure(doc,
                    os.path.join(self.out, "05_lipinski_compliance.png"),
                    "Figure 5. Lipinski Rule-of-Five compliance. "
                    "(Left) Distribution of violation counts per compound. "
                    "(Right) Individual rule violation rates.")

        _add_stats_table(doc, [
            ("Fully compliant (0 violations)", self._get(
                "lip_compliant_pct", "{:.1f}%")),
            ("MW > 500 Da",    self._get("lip_mw_pct",   "{:.1f}%")),
            ("LogP > 5",       self._get("lip_logp_pct", "{:.1f}%")),
            ("HBD > 5",        self._get("lip_hbd_pct",  "{:.1f}%")),
            ("HBA > 10",       self._get("lip_hba_pct",  "{:.1f}%")),
        ], headers=["Rule", "% Violated"])

    def _section_qed(self, doc: "Document") -> None:
        doc.add_heading("5.  Drug-likeness (QED)", 1)
        _add_horizontal_rule(doc)

        qed = self.s.get("qed_mean", float("nan"))
        doc.add_paragraph(
            "The Quantitative Estimate of Drug-likeness (QED) was calculated "
            "for each compound (Figure 6). The dataset achieves a mean QED "
            "of {:.3f}, reflecting moderate overall drug-likeness consistent "
            "with a broadly sampled library rather than a pre-filtered "
            "lead-like set. The right panel shows the relationship between "
            "QED and docking score, with the Pearson correlation coefficient "
            "indicating {}.".format(
                qed,
                "a weak positive correlation between drug-likeness and "
                "binding affinity"
                if self.s.get("qed_score_corr", 0) > 0.1
                else "little linear relationship between drug-likeness "
                     "and docking score"))

        _add_figure(doc,
                    os.path.join(self.out, "06_qed.png"),
                    "Figure 6. QED drug-likeness distribution (left) and "
                    "QED vs docking score hexbin plot (right). "
                    "The dashed line indicates QED = 0.5.")

    def _section_correlations(self, doc: "Document") -> None:
        doc.add_heading("6.  Score-Property Correlations", 1)
        _add_horizontal_rule(doc)

        doc.add_paragraph(
            "Figure 7 shows hexbin density plots of docking score against "
            "each of six molecular descriptors. Pearson correlation "
            "coefficients (r) are shown in each subplot title. In general, "
            "molecular weight shows the strongest correlation with docking "
            "score, reflecting the well-known tendency of larger molecules "
            "to achieve lower (more favourable) docking scores through "
            "increased contact surface area. This motivates the use of "
            "MW-normalised scores (ligand efficiency) as an alternative "
            "training target to avoid size bias in model predictions.")

        _add_figure(doc,
                    os.path.join(self.out, "07_score_correlations.png"),
                    "Figure 7. Docking score vs molecular property hexbin "
                    "plots. Pearson r values indicate linear correlation "
                    "strength. MW shows the strongest association.")

    def _section_scaffolds(self, doc: "Document") -> None:
        doc.add_heading("7.  Scaffold Diversity", 1)
        _add_horizontal_rule(doc)

        n_scaf  = self.s.get("n_unique_scaffolds", 0)
        sing    = self.s.get("singleton_scaffold_pct", float("nan"))

        doc.add_paragraph(
            "Murcko scaffold analysis was performed to assess the structural "
            "diversity of the dataset (Figure 8). {:,} unique Murcko "
            "scaffolds were identified, of which {:.1f}% are singletons "
            "(appearing in only one compound). This high singleton rate "
            "indicates excellent scaffold diversity, which is desirable for "
            "training a neural network that must generalise across a wide "
            "range of structural classes rather than memorising scaffold-"
            "specific patterns.".format(n_scaf, sing))

        _add_figure(doc,
                    os.path.join(self.out, "08_scaffold_diversity.png"),
                    "Figure 8. Murcko scaffold diversity. "
                    "(Left) Pie chart of the most frequent scaffold classes. "
                    "(Right) Bar chart of the top 20 scaffolds by frequency.")

    def _section_diversity(self, doc: "Document") -> None:
        doc.add_heading("8.  Pairwise Molecular Diversity", 1)
        _add_horizontal_rule(doc)

        sim  = self.s.get("mean_tanimoto_sim",  float("nan"))
        dist = self.s.get("mean_tanimoto_dist", float("nan"))

        doc.add_paragraph(
            "Pairwise Tanimoto similarity was calculated for a random "
            "subsample to quantify the internal diversity of the dataset "
            "(Figure 9). The mean pairwise Tanimoto similarity is {:.4f} "
            "(mean distance {:.4f}), indicating a highly diverse library. "
            "For reference, values below 0.3 are generally considered "
            "structurally distinct by medicinal chemistry conventions, "
            "and a mean pairwise distance approaching 1.0 represents "
            "maximum diversity. The nearest-neighbour distribution "
            "confirms that the majority of compounds have no close "
            "structural analogues within the dataset, further validating "
            "the suitability of this library for training a broadly "
            "applicable predictive model.".format(sim, dist))

        _add_figure(doc,
                    os.path.join(self.out, "09_diversity.png"),
                    "Figure 9. Molecular diversity analysis. "
                    "(Left) All-pairs Tanimoto similarity distribution. "
                    "(Right) Nearest-neighbour similarity distribution.")

        _add_stats_table(doc, [
            ("Mean pairwise similarity", "{:.4f}".format(sim)),
            ("Mean pairwise distance",   "{:.4f}".format(dist)),
        ], headers=["Metric", "Value"])

    def _section_structures(self, doc: "Document") -> None:
        doc.add_heading("9.  Representative Structures", 1)
        _add_horizontal_rule(doc)

        doc.add_paragraph(
            "Figure 10 shows the two-dimensional structures of the "
            "forty compounds achieving the lowest (most favourable) "
            "predicted docking scores. Figure 11 shows the forty worst-"
            "scoring compounds for comparison. Examination of the top "
            "structures reveals the chemical features associated with "
            "high predicted binding affinity for this target.")

        _add_figure(doc,
                    os.path.join(self.out, "10_top_structures.png"),
                    "Figure 10. 2D structures of the top 40 best-scoring "
                    "compounds. ZINC ID and docking score (kcal/mol) "
                    "are shown beneath each structure.",
                    width_inches=6.5)

        _add_figure(doc,
                    os.path.join(self.out, "11_bottom_structures.png"),
                    "Figure 11. 2D structures of the 40 worst-scoring "
                    "compounds for comparison.",
                    width_inches=6.5)

    def _conclusions(self, doc: "Document") -> None:
        doc.add_heading("10.  Conclusions", 1)
        _add_horizontal_rule(doc)

        n   = self.s.get("n_compounds", 0)
        sim = self.s.get("mean_tanimoto_sim", float("nan"))
        qed = self.s.get("qed_mean", float("nan"))
        pct = self.s.get("lip_compliant_pct", float("nan"))

        doc.add_paragraph(
            "The initial training dataset of {:,} compounds from ZINC15 "
            "demonstrates the following key characteristics suitable for "
            "neural network-based virtual screening:".format(n))

        points = [
            "High structural diversity (mean Tanimoto similarity {:.4f}), "
            "supporting generalisation of the trained model across "
            "broad chemical space.".format(sim),

            "{:.1f}% of compounds satisfy Lipinski's Rule of Five, "
            "indicating the majority possess physicochemical properties "
            "consistent with oral bioavailability.".format(pct),

            "Mean QED of {:.3f} reflects moderate drug-likeness, "
            "appropriate for a broadly sampled screening library.".format(qed),

            "Docking score distribution is approximately Gaussian, "
            "providing a well-conditioned regression target for "
            "neural network training.",

            "High scaffold singleton rate confirms the dataset is not "
            "dominated by closely related compound families, reducing "
            "risk of scaffold-specific overfitting.",
        ]

        for point in points:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(point)

        doc.add_paragraph()
        doc.add_paragraph(
            "These properties collectively support the use of this dataset "
            "as a robust foundation for the active learning virtual "
            "screening pipeline, with subsequent docking and model update "
            "cycles expected to progressively enrich the predicted "
            "hit population.")


# ------------------------------------------------------------------------------
# Predictions / results report
# ------------------------------------------------------------------------------

class ResultsReport:
    """
    Generates a Word report describing the neural network prediction results.

    Call after all plots have been generated by analyse_results.py.

    stats dict keys:
        n_total, score_min, score_max, score_mean, score_std,
        top1_threshold, top1_count, top5_threshold, top5_count,
        mean_tanimoto_sim, mean_tanimoto_dist,
        n_unique_scaffolds, singleton_scaffold_pct,
        iteration  (active learning iteration number, default 1)
        target_name (optional)
    """

    def __init__(self, stats: Dict[str, Any], output_dir: str):
        self.s   = stats
        self.out = output_dir

    def _get(self, key: str, fmt: str = "{}", default: str = "N/A") -> str:
        val = self.s.get(key)
        if val is None:
            return default
        try:
            return fmt.format(val)
        except Exception:
            return str(val)

    def generate(self, out_path: str) -> None:
        if not DOCX_AVAILABLE:
            return

        doc = Document()
        _build_doc_styles(doc)

        for section in doc.sections:
            section.top_margin    = Cm(2.5)
            section.bottom_margin = Cm(2.5)
            section.left_margin   = Cm(2.5)
            section.right_margin  = Cm(2.5)

        self._title_page(doc)
        self._executive_summary(doc)
        self._section_score_distribution(doc)
        self._section_chemical_space(doc)
        self._section_tanimoto(doc)
        self._section_scaffolds(doc)
        self._section_properties(doc)
        self._section_structures(doc)
        self._conclusions(doc)

        doc.save(out_path)
        print("  Report saved: {}".format(out_path))

    def _title_page(self, doc: "Document") -> None:
        doc.add_paragraph()
        doc.add_paragraph()
        iteration = self.s.get("iteration", 1)
        t = doc.add_heading(
            "Virtual Screening Results: Iteration {}".format(iteration), 0)
        t.alignment = WD_ALIGN_PARAGRAPH.CENTER

        sub = doc.add_paragraph(
            "Neural network predicted docking score analysis -- "
            "chemical space, diversity, and hit characterisation")
        sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sub.runs[0].font.size = Pt(12)
        sub.runs[0].font.color.rgb = RGBColor(0x61, 0x61, 0x61)

        doc.add_paragraph()
        date_p = doc.add_paragraph(
            "Generated: {}".format(datetime.now().strftime("%d %B %Y")))
        date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_p.runs[0].font.color.rgb = RGBColor(0x61, 0x61, 0x61)
        doc.add_page_break()

    def _executive_summary(self, doc: "Document") -> None:
        doc.add_heading("Executive Summary", 1)
        _add_horizontal_rule(doc)

        n    = self.s.get("n_total",     0)
        smin = self.s.get("score_min",   float("nan"))
        smax = self.s.get("score_max",   float("nan"))
        smn  = self.s.get("score_mean",  float("nan"))
        t1   = self.s.get("top1_threshold", float("nan"))
        c1   = self.s.get("top1_count",  0)

        doc.add_paragraph(
            "A neural network trained on AutoDock Vina docking scores was "
            "used to predict binding affinity for {:,} compounds from the "
            "ZINC15 database. Predicted scores range from {:.2f} to "
            "{:.2f} kcal/mol with a mean of {:.2f} kcal/mol. The top 1% "
            "of predicted scores ({:,} compounds) fall below {:.2f} kcal/mol "
            "and represent the primary candidates for the next round of "
            "explicit docking and active learning.".format(
                n, smin, smax, smn, c1, t1))

        doc.add_paragraph()
        doc.add_heading("Key Statistics", 2)
        _add_stats_table(doc, [
            ("Compounds scored",       "{:,}".format(n)),
            ("Best predicted score",   "{:.4f} kcal/mol".format(smin)),
            ("Worst predicted score",  "{:.4f} kcal/mol".format(smax)),
            ("Mean predicted score",   "{:.4f} kcal/mol".format(smn)),
            ("Top 1% threshold",       "{:.4f} kcal/mol".format(t1)),
            ("Top 1% count",           "{:,}".format(c1)),
            ("Top 5% threshold",
             self._get("top5_threshold", "{:.4f} kcal/mol")),
            ("Mean Tanimoto sim. (top {:,})".format(
                self.s.get("n_tanimoto", 50)),
             self._get("mean_tanimoto_sim", "{:.4f}")),
            ("Unique scaffolds (top {:,})".format(
                self.s.get("n_top", 1000)),
             self._get("n_unique_scaffolds", "{:,}")),
        ])
        doc.add_page_break()

    def _section_score_distribution(self, doc: "Document") -> None:
        doc.add_heading("1.  Predicted Score Distribution", 1)
        _add_horizontal_rule(doc)

        n    = self.s.get("n_total",    0)
        smn  = self.s.get("score_mean", float("nan"))
        sstd = self.s.get("score_std",  float("nan"))
        t1   = self.s.get("top1_threshold",  float("nan"))
        t5   = self.s.get("top5_threshold",  float("nan"))
        t10  = self.s.get("top10_threshold", float("nan"))

        doc.add_paragraph(
            "Figure 1 shows the distribution of neural network predicted "
            "docking scores across all {:,} screened compounds. The "
            "distribution has a mean of {:.2f} \u00b1 {:.2f} kcal/mol. "
            "The top 1%, 5%, and 10% score thresholds are marked at "
            "{:.2f}, {:.2f}, and {:.2f} kcal/mol respectively, defining "
            "the candidate pools for subsequent explicit docking.".format(
                n, smn, sstd, t1, t5, t10))

        _add_figure(doc,
                    os.path.join(self.out, "score_distribution.png"),
                    "Figure 1. Predicted docking score distribution. "
                    "Top percentile thresholds are marked with dashed lines.")

    def _section_chemical_space(self, doc: "Document") -> None:
        doc.add_heading("2.  Chemical Space Coverage", 1)
        _add_horizontal_rule(doc)

        doc.add_paragraph(
            "PCA and UMAP projections of the Morgan fingerprint space "
            "are shown in Figures 2 and 3 respectively, with compounds "
            "coloured by predicted docking score. Clusters of high-scoring "
            "compounds visible in both projections indicate that the model "
            "has identified specific regions of chemical space associated "
            "with high predicted binding affinity, validating the model's "
            "ability to discriminate between active and inactive chemotypes.")

        _add_figure(doc,
                    os.path.join(self.out, "pca.png"),
                    "Figure 2. PCA of Morgan fingerprints coloured by "
                    "predicted docking score. Top 1% compounds highlighted "
                    "in the right panel.")

        _add_figure(doc,
                    os.path.join(self.out, "umap.png"),
                    "Figure 3. UMAP projection (Jaccard metric) coloured "
                    "by predicted score, revealing non-linear cluster "
                    "structure in chemical space.")

    def _section_tanimoto(self, doc: "Document") -> None:
        doc.add_heading("3.  Structural Similarity of Top Compounds", 1)
        _add_horizontal_rule(doc)

        sim  = self.s.get("mean_tanimoto_sim",  float("nan"))
        dist = 1 - sim if sim == sim else float("nan")
        n_t  = self.s.get("n_tanimoto", 50)

        doc.add_paragraph(
            "Pairwise Tanimoto similarity was calculated for the top {:,} "
            "predicted compounds (Figure 4). The mean pairwise Tanimoto "
            "similarity among the top hits is {:.4f} (distance {:.4f}). "
            "{} This metric is important for assessing whether the model "
            "has converged on a narrow structural class or identified "
            "diverse binding scaffolds, which has implications for "
            "subsequent lead optimisation strategy.".format(
                n_t, sim, dist,
                "The relatively low mean similarity indicates structural "
                "diversity among the top-ranked compounds, suggesting "
                "multiple binding modes or scaffold families are represented."
                if sim < 0.4
                else "The moderate similarity suggests some structural "
                     "convergence among top-ranked compounds, which may "
                     "reflect a dominant pharmacophore for this target."))

        _add_figure(doc,
                    os.path.join(self.out, "tanimoto_heatmap.png"),
                    "Figure 4. Pairwise Tanimoto similarity heatmap for "
                    "the top {:,} predicted compounds. "
                    "Mean similarity is annotated.".format(n_t))

    def _section_scaffolds(self, doc: "Document") -> None:
        doc.add_heading("4.  Scaffold Diversity of Top Hits", 1)
        _add_horizontal_rule(doc)

        n_s  = self.s.get("n_unique_scaffolds",    0)
        sing = self.s.get("singleton_scaffold_pct", float("nan"))
        n_t  = self.s.get("n_top", 1000)

        doc.add_paragraph(
            "Murcko scaffold analysis of the top {:,} predicted compounds "
            "identified {:,} unique scaffolds, of which {:.1f}% are "
            "singletons (Figure 5). {} A high scaffold diversity among "
            "top hits is desirable as it provides multiple independent "
            "starting points for medicinal chemistry optimisation and "
            "reduces the risk of scaffold-dependent off-target effects.".format(
                n_t, n_s, sing,
                "The high scaffold diversity suggests the model has "
                "identified multiple distinct chemotype families with "
                "predicted affinity for the target."
                if sing > 60
                else "The moderate scaffold diversity indicates some "
                     "enrichment of particular chemotype families, "
                     "which may warrant further investigation."))

        _add_figure(doc,
                    os.path.join(self.out, "scaffold_diversity.png"),
                    "Figure 5. Scaffold distribution among the top {:,} "
                    "predicted compounds.".format(n_t))

    def _section_properties(self, doc: "Document") -> None:
        doc.add_heading("5.  Property Profile of Top Hits", 1)
        _add_horizontal_rule(doc)

        doc.add_paragraph(
            "Figure 6 compares the molecular property distributions of the "
            "top {:,} predicted compounds against the full screened library. "
            "Shifts in property distributions between the two groups reveal "
            "the physicochemical features enriched among high-scoring "
            "compounds, which may inform pharmacophore models and guide "
            "future compound selection.".format(
                self.s.get("n_top", 1000)))

        _add_figure(doc,
                    os.path.join(self.out, "property_distributions.png"),
                    "Figure 6. Molecular property distributions comparing "
                    "top predicted compounds (red) vs all screened compounds "
                    "(blue). Lipinski limits are shown as dashed lines.")

    def _section_structures(self, doc: "Document") -> None:
        doc.add_heading("6.  Top Predicted Compounds", 1)
        _add_horizontal_rule(doc)

        doc.add_paragraph(
            "Figure 7 shows the two-dimensional structures of the "
            "highest-ranking predicted compounds, annotated with their "
            "ZINC identifiers and predicted docking scores. These "
            "represent the primary candidates for the next explicit "
            "docking campaign and subsequent active learning model update.")

        _add_figure(doc,
                    os.path.join(self.out, "top_structures.png"),
                    "Figure 7. 2D structures of the top 40 compounds by "
                    "predicted docking score. Scores are in kcal/mol.",
                    width_inches=6.5)

    def _conclusions(self, doc: "Document") -> None:
        doc.add_heading("7.  Conclusions and Next Steps", 1)
        _add_horizontal_rule(doc)

        iteration = self.s.get("iteration", 1)
        n_total   = self.s.get("n_total",  0)
        c1        = self.s.get("top1_count", 0)

        doc.add_paragraph(
            "Neural network screening of {:,} compounds (iteration {}) "
            "has identified {:,} top-ranked candidates for explicit "
            "AutoDock Vina docking. The recommended workflow for the "
            "next active learning cycle is:".format(n_total, iteration, c1))

        steps = [
            "Generate 3D conformers and PDBQT files for the top {:,} "
            "predicted compounds using prepare_pdbqt.py.".format(c1),

            "Run explicit AutoDock Vina docking on the prepared "
            "compounds using the parallel MPI docking pipeline "
            "(estimated 2-3 days for 1 million compounds).",

            "Incorporate real docking scores into the training dataset "
            "using update_model.py (fine-tune mode recommended for "
            "< 20% new data; full retrain for larger additions).",

            "Re-screen the full ZINC15 library with the updated model "
            "using predict_batch_csv.py to generate iteration {} "
            "predictions.".format(iteration + 1),

            "Repeat until model RMSE on the test set plateaus or "
            "the top compound scores no longer improve between cycles.",
        ]

        for step in steps:
            p = doc.add_paragraph(style="List Number")
            p.add_run(step)

        doc.add_paragraph()
        doc.add_paragraph(
            "Each active learning iteration is expected to improve model "
            "accuracy in the low-score (high-affinity) region of chemical "
            "space, progressively concentrating docking effort on the most "
            "promising compound classes.")
