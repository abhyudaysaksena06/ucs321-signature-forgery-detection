"""Report 1 and the notebook's closing text, all numbers read from results.json.
Usage: build_report.py rundir outdir"""
import json, sys
from pathlib import Path
from PIL import Image as PILImage
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

run, out = Path(sys.argv[1]), Path(sys.argv[2])
R = json.loads((run / "results.json").read_text())
T = R["table"]
REF, SIN, SVM, RAW = T["Reference CNN"], T["Single-image CNN"], T["Reference SVM"], T["Raw-image control (invalid)"]
RF, RBF, LR = T["Random Forest"], T["SVM (RBF)"], T["Logistic Regression"]
pc = lambda v: f"{v * 100:.1f}%"
cm = R["confusion_matrix"]
tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
n = R["test_queries"]
pw = R["per_writer"]; lo, hi = min(pw.values()), max(pw.values())
acc_rows = R["stability"]
acc_vals = [v["Accuracy"] for v in acc_rows.values()]
gap = (R["leak_acc"] - R["single_acc"]) * 100

TITLE = "Detecting Forged Handwritten Signatures with a Reference-Based Convolutional Neural Network"
BYLINE = "UCS321 AI for Engineers, Report 1 | Akshara Thapan | Roll No. 7025000032 | Topic T-178"

FIG1 = (f"Figure 1: Background grey level (left) separates the classes completely across {R['n_images']:,} images, while ink coverage (right) overlaps heavily.")
FIG2 = ("Figure 2: Reference CNN on test signers: confusion matrix (left), ROC curves with AUC in legend (centre) and accuracy by signer (right).")
TAB1 = (f"Table 1: Test results on the same {n} signatures from eight unseen signers, with forged as the positive class. "
        "FAR is the share of forgeries accepted and FRR the share of genuine signatures rejected.")

SECTIONS = [
("1. Introduction", [
 "Banks, courts and offices still accept handwritten signatures, and a human examiner checking them is slow and inconsistent. "
 "A model that flags likely forgeries could direct that attention where it is needed. My allocated topic, detecting forged handwritten "
 "signatures, is a binary image classification task: given a signature, decide whether it is genuine or a forgery. I made one requirement "
 "central from the start: the system must work on people it has never seen, because a bank cannot retrain a model for every new customer.",
]),
("2. Dataset", [
 f"I used the CEDAR signature database [1]: {R['n_writers']} signers, each with 24 genuine signatures and 24 skilled forgeries, scanned at 300 dpi. "
 f"The notebook loads a public GitHub copy pinned to a single commit [2], because the official file is a RAR archive. The copy held {R['n_images'] + R['n_extra_removed']:,} files, "
 f"of which {R['n_extra_removed']} were stray test files that were exact copies of other images, so I removed them, leaving {R['n_images']:,}. "
 "The features are the pixels and the target is genuine (0) or forged (1). Every signer contributes the same number of each class, so random guessing scores 50% and accuracy is a fair metric.",
 f"The most important decision in this project came from a routine check. Before training, I measured each image's background brightness. "
 f"Figure 1 shows that genuine images have a median grey level near {R['background_medians']['genuine']:.0f} and forged images near {R['background_medians']['forged']:.0f}, with no overlap. "
 f"This has nothing to do with handwriting: the two groups appear to have been scanned or processed differently. A one-line rule using brightness alone scored {pc(R['background_rule_acc'])} on my test set, "
 "without looking at a single stroke. The real difficulty is closer to the right panel, where ink coverage barely separates the classes.",
]),
("FIG1", []),
("3. Methodology", [
 "Pre-processing was therefore designed around removing this shortcut. I applied Otsu's threshold [4] to each image to keep only the ink as a black-and-white mask, cropped to the signature and "
 "resized it to 96 × 192 pixels with padding so that its shape was kept. Every background is now identical. The grey version was kept only as a control. "
 f"The IQR rule flagged {R['outliers_aspect']} images on aspect ratio and {R['outliers_ink']} on ink coverage. I kept them, since long or sparse signatures are real, and the padding handles their shape.",
 "I split the data by signer rather than by image: 39 signers for training, 8 for validation and 8 for testing, so nobody appears in two sets. "
 "I chose this because a random image split would test the model on people it had already met. To measure the effect, I also retrained one model on a random split.",
 "The topic names a CNN, so I built three from the same four convolutional blocks. The single-image CNN sees only the signature in question. The Reference CNN, my main model, receives that signature together "
 "with a genuine signature of the same person as a two-channel image, and its prediction is averaged over five enrolled references. I chose this design because real verification compares against a specimen on file, "
 "and the idea follows the writer-independent approach of SigNet [3]. A twin (Siamese) network was also tried, but it was slower and sat near 60% after five epochs in an early trial, so I stopped it. The third CNN used raw images, only to measure the artefact. "
 "Training used Adam with a cosine learning-rate schedule, dropout and small random rotations, scalings and shifts, because nobody signs identically twice. The best epoch was chosen on validation accuracy. "
 "For comparison I used Logistic Regression, an RBF SVM and a Random Forest on shrunken images, plus a Reference SVM that receives the same reference information as the Reference CNN. These were implemented in PyTorch and scikit-learn [5].",
]),
("4. Results", [
 f"Table 1 shows that the Reference CNN was the best valid model, with {pc(REF['Accuracy'])} accuracy ({tn + tp} of {n}) and an ROC AUC of {REF['ROC AUC']:.3f}. "
 f"This is slightly under the expected 80%. The single-image CNN reached {pc(SIN['Accuracy'])}, so seeing a reference added {(REF['Accuracy'] - SIN['Accuracy']) * 100:.1f} points. The classical models stayed between {pc(LR['Accuracy'])} and {pc(RF['Accuracy'])}, "
 f"and the Reference SVM reached only {pc(SVM['Accuracy'])} despite the same input, which suggests that comparing raw pixels is not enough and that the convolutional filters are doing the useful work. "
 f"The raw-image control scored {pc(RAW['Accuracy'])}. I report it but treat it as invalid, since Figure 1 explains how it was achieved.",
]),
("TAB1", []),
("Results2", [
 f"The confusion matrix in Figure 2 shows that errors are fairly balanced: {fn} forgeries were accepted (FAR {pc(REF['FAR'])}) and {fp} genuine signatures rejected (FRR {pc(REF['FRR'])}). "
 f"The AUC of {REF['ROC AUC']:.3f} means the probabilities rank signatures better than the 0.50 cut-off suggests, so a bank could lower the cut-off to catch more forgeries at the cost of rejecting more customers. "
 f"Results differ widely by person, from {pc(lo)} to {pc(hi)} across the eight test signers, so some signatures are clearly easier to verify than others.",
]),
("FIG2", []),
("Results3", [
 f"Two further checks test these results. Repeating the Reference CNN on three different signer splits gave accuracies of {', '.join(pc(v) for v in acc_vals)} (mean {pc(R['stability_mean'])}, standard deviation {R['stability_std'] * 100:.1f} points), so the choice of signers matters. "
 f"For the leakage check, the single-image CNN scored {pc(R['leak_acc'])} on a random image split against {pc(R['single_acc'])} on the signer split, a difference of {gap:.1f} points.",
]),
("5. Discussion and Conclusion", [
 f"The best model is the Reference CNN, and my main finding is about method rather than architecture. A careless pipeline would have reported {pc(RAW['Accuracy'])}, and that number would have been meaningless. "
 f"After the shortcut was removed, the honest figure is {pc(REF['Accuracy'])} on strangers. It falls just short of 80%, and I believe the reasons are real: only 39 training signers, skilled forgeries written to imitate the target, and the removal of the easiest cue. "
 "The noisy validation curves also suggest that the model had not fully settled. I did not keep adjusting the model against the test set to pass 80%, because that would make the test score unreliable.",
 "There are clear limitations. Eight test signers is a small sample, and binarising also discards pen-pressure information that may be useful. The artefact suggests each class came from its own scanning setup, so these results may not transfer to other scanners. "
 "With more time, I would try a pretrained network, which I did not test here, train on a larger multi-writer dataset, and choose the cut-off from the real cost of a missed forgery. The wider lesson for me is that a perfect score should be investigated, not celebrated.",
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
TAB1 += " *Invalid: it reads the scanner artefact."

# ---------------------------------------------------------------- PDF
F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("TNR", F + "times.ttf"))
pdfmetrics.registerFont(TTFont("TNR-B", F + "timesbd.ttf"))
LEAD = 20.7   # 1.5 lines of 12 pt Times New Roman, as Word computes it
body = ParagraphStyle("b", fontName="TNR", fontSize=12, leading=LEAD, alignment=TA_JUSTIFY, spaceAfter=0)
head = ParagraphStyle("h", parent=body, fontName="TNR-B", keepWithNext=1, spaceBefore=2)
cap = ParagraphStyle("c", parent=body)

def figure(name, width):
    path = run / "figures" / name
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)

W = A4[0] - 2 * inch
story = [Paragraph(TITLE, head), Paragraph(BYLINE, body)]
for title, paras in SECTIONS:
    if title == "FIG1":
        story.append(KeepTogether([figure("background_artefact.png", 370), Paragraph(FIG1, cap)]))
    elif title == "FIG2":
        story.append(KeepTogether([figure("evaluation.png", 451), Paragraph(FIG2, cap)]))
    elif title == "TAB1":
        t = Table(table_data, colWidths=[W - 7 * 44] + [44] * 7, rowHeights=[17] * len(table_data))
        t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "TNR", 12), ("FONT", (0, 0), (-1, 0), "TNR-B", 12), ("FONT", (0, 1), (-1, 1), "TNR-B", 12),
                               ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                               ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                               ("LINEABOVE", (0, 0), (-1, 0), .8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), .5, colors.black), ("LINEBELOW", (0, -1), (-1, -1), .8, colors.black)]))
        story.append(KeepTogether([Paragraph(TAB1, cap), t, Spacer(1, 6)]))
    else:
        if not title.startswith("Results"):
            story.append(Paragraph(title, head))
        story += [Paragraph(p, body) for p in paras]
story.append(PageBreak())
story.append(Paragraph("References", head))
for r in REFS:
    story.append(Paragraph(r, body))

stem = "7025000032_AksharaThapan_Report1"
pdf = SimpleDocTemplate(str(out / f"{stem}.pdf"), pagesize=A4, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch,
                        title=TITLE, author="Akshara Thapan")
pdf.build(story)

# ---------------------------------------------------------------- Word copy, for editing
doc = Document()
s = doc.sections[0]; s.page_width, s.page_height = Inches(8.27), Inches(11.69)
s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.line_spacing = 1.5; st.paragraph_format.space_after = Pt(0)
def para(text, bold=False):
    p = doc.add_paragraph(); r = p.add_run(text); r.bold = bold; p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; return p
para(TITLE, True); para(BYLINE)
for title, paras in SECTIONS:
    if title == "FIG1":
        doc.add_picture(str(run / "figures/background_artefact.png"), width=Inches(6.27)); para(FIG1)
    elif title == "FIG2":
        doc.add_picture(str(run / "figures/evaluation.png"), width=Inches(6.27)); para(FIG2)
    elif title == "TAB1":
        para(TAB1)
        t = doc.add_table(rows=len(table_data), cols=len(HEAD)); t.style = "Table Grid"
        for i, row in enumerate(table_data):
            for j, v in enumerate(row):
                c = t.cell(i, j); c.text = v; c.paragraphs[0].paragraph_format.line_spacing = 1
    else:
        if not title.startswith("Results"):
            para(title, True)
        for p in paras:
            para(p)
doc.add_page_break(); para("References", True)
for r in REFS:
    para(r)
doc.save(out / f"{stem}.docx")

# ---------------------------------------------------------------- notebook closing text
justification = (
    f"The Reference CNN reached **{pc(REF['Accuracy'])}** on unseen signers, just below the expected 80%, so here is why.\n\n"
    f"**Why it is below 80%:**\n"
    f"- **The scanner shortcut was removed on purpose.** Raw images scored {pc(RAW['Accuracy'])}, but a one-line brightness rule scored {pc(R['background_rule_acc'])} too, so that number says nothing about handwriting.\n"
    "- **Small dataset.** Only 39 people are used for training, and the model must generalise to new people.\n"
    "- **Very similar classes.** The forgeries are skilled imitations of the same signature.\n"
    "- **Unstable learning.** The validation curves in Section 5 are noisy, which shows the model has not fully settled.\n\n"
    "**What I tried, and what happened:**\n"
    f"- Augmentation (rotation, scale, shift): included in all final models.\n"
    f"- A different question: adding a genuine reference raised accuracy from {pc(SIN['Accuracy'])} to {pc(REF['Accuracy'])}.\n"
    "- A different architecture: a twin (Siamese) network was slower and sat near 60% after five epochs in an early trial, so I stopped it.\n"
    "- More stable training: validation accuracy swung widely without a learning-rate schedule, so I added a cosine schedule, which helped only partly.\n"
    "- A pretrained model: **not tried**, and it is the next step.\n\n"
    "I did not keep adjusting the model against the test set to pass 80%, because the test score would then stop being a fair estimate."
)
conclusion = (
    f"The best valid model is the **Reference CNN**, with {pc(REF['Accuracy'])} accuracy, F1 {REF['F1']:.3f} and ROC AUC {REF['ROC AUC']:.3f} on signers it never saw.\n"
    f"It beat the single-image CNN ({pc(SIN['Accuracy'])}) and every classical model (best {pc(RF['Accuracy'])}), so comparing against a genuine reference is what helped most.\n"
    f"The key finding is that the raw-image score of {pc(RAW['Accuracy'])} is false: genuine and forged images were scanned differently, and brightness alone separates them.\n"
    f"A random image split would also have added {gap:.1f} points to the single-image CNN, so splitting by signer was essential.\n"
    f"Over three signer splits the accuracy averaged {pc(R['stability_mean'])} with a standard deviation of {R['stability_std'] * 100:.1f} points, and it ranged from {pc(lo)} to {pc(hi)} across test signers.\n"
    "Limitations: only 8 test signers, 39 training signers, and an apparent difference in scanning between the classes, so these results may not hold on other scanners."
)
(run / "text.json").write_text(json.dumps({"justification": justification, "conclusion": conclusion}, ensure_ascii=False, indent=1), encoding="utf-8")
print("built", stem)
