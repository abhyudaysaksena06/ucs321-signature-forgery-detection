"""Shorter, formal version of Report 1, all numbers read from results.json.
Usage: build_formal_report.py results_dir out_dir"""
import json, sys
from pathlib import Path
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

src, out = Path(sys.argv[1]), Path(sys.argv[2])
R = json.loads((src / "results.json").read_text())
T = R["table"]
REF, SIN, SVM, RAW = T["Reference CNN"], T["Single-image CNN"], T["Reference SVM"], T["Raw-image control (invalid)"]
RF, RBF, LR = T["Random Forest"], T["SVM (RBF)"], T["Logistic Regression"]
pc = lambda v: f"{v * 100:.1f}%"
(tn, fp), (fn, tp) = R["confusion_matrix"]
n = R["test_queries"]
pw = R["per_writer"]
gap = (R["leak_acc"] - R["single_acc"]) * 100

TITLE = "Detection of Forged Handwritten Signatures Using a Reference-Based Convolutional Neural Network"
BYLINE = ["Akshara Thapan (Roll No. 7025000032)", "UCS321 AI for Engineers, Report 1, Topic T-178"]

FIG1 = (f"Figure 1: Background grey level (left) separates the classes completely across {R['n_images']:,} images; "
        "ink coverage (right) overlaps heavily.")
FIG2 = ("Figure 2: Reference CNN on the test signers: confusion matrix (left), ROC curves with AUC in the legend (centre) "
        "and accuracy by signer (right).")
TAB1 = (f"Table 1: Results on {n} signatures from unseen signers (forged is positive). "
        "FAR: forgeries accepted. FRR: genuine rejected. *Invalid: uses the artefact.")

SECTIONS = [
("Abstract", [
 f"This report presents a convolutional neural network (CNN) for detecting forged handwritten signatures in the CEDAR database. "
 f"Analysis of the raw scans revealed a background artefact that permits perfect classification without reference to the handwriting; this was removed by binarisation. "
 f"Evaluated on eight signers excluded from training, a CNN that compares each signature with a genuine reference achieved an accuracy of {pc(REF['Accuracy'])} and a ROC AUC of {REF['ROC AUC']:.3f}, "
 "outperforming a single-image CNN and all classical baselines.",
]),
("1. Introduction", [
 "Handwritten signatures remain a common means of authentication in banking and legal practice, yet manual verification is slow and subjective. "
 "The objective of this work is a binary classifier that separates genuine signatures from skilled forgeries. "
 "Because a deployed system must handle new customers without retraining, all evaluation was performed on signers absent from training.",
]),
("2. Dataset", [
 f"The CEDAR signature database [1] contains {R['n_writers']} signers, each with 24 genuine signatures and 24 skilled forgeries. A public GitHub copy pinned to one commit was used [2]. "
 f"Six stray files that duplicated other images were removed, leaving {R['n_images']:,} images. The classes are balanced, so chance accuracy is 50%.",
 f"An initial data check found that the median background grey level was near {R['background_medians']['genuine']:.0f} for genuine images and {R['background_medians']['forged']:.0f} for forged images, with no overlap (Figure 1). "
 f"A single brightness threshold therefore achieved {pc(R['background_rule_acc'])} test accuracy without examining any handwriting. This indicates a difference in scanning, not a property of the signatures, and it had to be removed.",
]),
("3. Methodology", [
 f"Each image was binarised with Otsu's method [4], cropped to the signature and resized to 96 × 192 pixels with padding, which gives every image an identical background. "
 f"The IQR rule flagged {R['outliers_aspect']} images on aspect ratio and {R['outliers_ink']} on ink coverage; these were retained as legitimate variation. "
 "The data were split by signer into 39 training, 8 validation and 8 test signers, so that no individual appears in more than one set.",
]),
("FIG1", []),
("Results-method", [
 "Three CNNs were built from the same four convolutional blocks. The single-image CNN receives only the questioned signature. The Reference CNN, the principal model, receives the questioned signature and a genuine signature of the same person "
 "as a two-channel input, reflecting the comparison made in practice and the writer-independent approach of SigNet [3]; its output is averaged over five enrolled references. A third CNN, trained on raw images, served only as a control. "
 "Training used Adam with a cosine learning-rate schedule, dropout and light geometric augmentation, and the best epoch was selected on validation accuracy. "
 "Logistic Regression, an RBF SVM, a Random Forest and a Reference SVM were used for comparison [5].",
]),
("4. Results", [
 f"Table 1 shows that the Reference CNN was the best valid model, correctly classifying {tn + tp} of {n} test signatures ({pc(REF['Accuracy'])}), slightly below the expected 80%. "
 f"Access to a reference improved accuracy by {(REF['Accuracy'] - SIN['Accuracy']) * 100:.1f} points over the single-image CNN. The Reference SVM reached only {pc(SVM['Accuracy'])} with equivalent input, "
 "which suggests that the convolutional filters contribute substantially to the comparison.",
]),
("TAB1", []),
("Results2", [
 f"Errors were reasonably balanced, with {fn} forgeries accepted and {fp} genuine signatures rejected (Figure 2). Accuracy ranged from {pc(min(pw.values()))} to {pc(max(pw.values()))} across the test signers. "
 f"Repeating training on three signer splits gave a mean accuracy of {pc(R['stability_mean'])} with a standard deviation of {R['stability_std'] * 100:.1f} points. "
 f"A random image split raised the single-image CNN from {pc(R['single_acc'])} to {pc(R['leak_acc'])}, showing that {gap:.1f} points would have been a product of leakage.",
]),
("FIG2", []),
("5. Discussion and Conclusion", [
 f"The Reference CNN is the recommended model. The principal finding concerns method: the {pc(RAW['Accuracy'])} obtained from raw images is not meaningful, and the valid result on unseen signers is {pc(REF['Accuracy'])}. "
 "The shortfall from 80% is attributed to the small number of training signers, the similarity of skilled forgeries to genuine signatures, and the removal of the artefact. The model was not tuned further against the test set, as this would invalidate the estimate.",
 "The study is limited by its eight test signers and by the possibility that the results do not transfer to other scanners. Binarisation also discards pen-pressure information. "
 "Future work should evaluate a pretrained network, use a larger multi-writer dataset, and set the decision threshold according to the cost of an accepted forgery.",
]),
]

REFS = [
 "[1] M. K. Kalera, S. Srihari and A. Xu, “Offline signature verification and identification using distance statistics,” Int. J. Pattern Recognition and Artificial Intelligence, vol. 18, no. 7, pp. 1339–1360, 2004.",
 "[2] CEDAR signature images. Official: https://cedar.buffalo.edu/NIJ/data/signatures.rar. GitHub copy used: https://github.com/Snehal-Aparadh/SignatureForgeryDetection (commit 9e4157b).",
 "[3] S. Dey et al., “SigNet: Convolutional Siamese network for writer independent offline signature verification,” arXiv:1707.02131, 2017.",
 "[4] N. Otsu, “A threshold selection method from gray-level histograms,” IEEE Trans. Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62–66, 1979.",
 "[5] PyTorch, https://pytorch.org; F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
 "AI tool use: Claude (Anthropic) was used to help design the experiments, write the notebook code and draft this report.",
]

ROWS = [("Reference CNN", REF), ("Single-image CNN", SIN), ("Random Forest", RF), ("Reference SVM", SVM), ("SVM (RBF)", RBF), ("Logistic Regression", LR), ("Raw control*", RAW)]
HEAD = ["Model", "Acc.", "Prec.", "Rec.", "F1", "AUC", "FAR", "FRR"]
KEYS = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC", "FAR", "FRR"]
table_data = [HEAD] + [[name] + [f"{m[k]:.3f}" for k in KEYS] for name, m in ROWS]

F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("TNR", F + "times.ttf"))
pdfmetrics.registerFont(TTFont("TNR-B", F + "timesbd.ttf"))
LEAD = 20.7
body = ParagraphStyle("b", fontName="TNR", fontSize=12, leading=LEAD, alignment=TA_JUSTIFY)
head = ParagraphStyle("h", parent=body, fontName="TNR-B", spaceBefore=2)
title = ParagraphStyle("t", parent=head, alignment=TA_CENTER)
centre = ParagraphStyle("m", parent=body, alignment=TA_CENTER)

def figure(name, width):
    path = src / "figures" / name
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)

W = A4[0] - 2 * inch
story = [Paragraph(TITLE, title)] + [Paragraph(b, centre) for b in BYLINE]
for name, paras in SECTIONS:
    if name == "FIG1":
        story.append(KeepTogether([figure("background_artefact.png", 370), Paragraph(FIG1, body)]))
    elif name == "FIG2":
        story.append(KeepTogether([figure("evaluation.png", W), Paragraph(FIG2, body)]))
    elif name == "TAB1":
        t = Table(table_data, colWidths=[W - 7 * 44] + [44] * 7, rowHeights=[17] * len(table_data))
        t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "TNR", 12), ("FONT", (0, 0), (-1, 0), "TNR-B", 12), ("FONT", (0, 1), (-1, 1), "TNR-B", 12),
                               ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                               ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                               ("LINEABOVE", (0, 0), (-1, 0), .8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), .5, colors.black), ("LINEBELOW", (0, -1), (-1, -1), .8, colors.black)]))
        story.append(KeepTogether([Paragraph(TAB1, body), t, Spacer(1, 6)]))
    else:
        if not name.startswith("Results"):
            story.append(Paragraph(name, head))
        story += [Paragraph(p, body) for p in paras]
story.append(PageBreak())
story.append(Paragraph("References", head))
story += [Paragraph(r, body) for r in REFS]

stem = "7025000032_AksharaThapan_Report1_Formal"
SimpleDocTemplate(str(out / f"{stem}.pdf"), pagesize=A4, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch,
                  title=TITLE, author="Akshara Thapan").build(story)

doc = Document()
s = doc.sections[0]; s.page_width, s.page_height = Inches(8.27), Inches(11.69)
s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.line_spacing = 1.5; st.paragraph_format.space_after = Pt(0)
def para(text, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph(); r = p.add_run(text); r.bold = bold; p.alignment = align; return p
para(TITLE, True, WD_ALIGN_PARAGRAPH.CENTER)
for b in BYLINE:
    para(b, align=WD_ALIGN_PARAGRAPH.CENTER)
for name, paras in SECTIONS:
    if name == "FIG1":
        doc.add_picture(str(src / "figures/background_artefact.png"), width=Inches(5.14)); para(FIG1)
    elif name == "FIG2":
        doc.add_picture(str(src / "figures/evaluation.png"), width=Inches(6.27)); para(FIG2)
    elif name == "TAB1":
        para(TAB1)
        t = doc.add_table(rows=len(table_data), cols=len(HEAD)); t.style = "Table Grid"
        for i, row in enumerate(table_data):
            for j, v in enumerate(row):
                c = t.cell(i, j); c.text = v; c.paragraphs[0].paragraph_format.line_spacing = 1
    else:
        if not name.startswith("Results"):
            para(name, True)
        for p in paras:
            para(p)
doc.add_page_break(); para("References", True)
for r in REFS:
    para(r)
doc.save(out / f"{stem}.docx")
print("built", stem)
