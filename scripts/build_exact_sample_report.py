"""
build_exact_sample_report.py - Builds the Two-Page Report replicating the exact
visual layout, fonts (Times-Roman), tables, figures, and styling of the user's sample document.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            super().showPage()
        super().save()


def build_pdf():
    pdf_path = Path("D:/CBL/reports/PS21_Kidney_Abnormality_Detection_Report.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # 0.55 in (40 pt) margins matching the reference document
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=44,
        rightMargin=44,
        topMargin=38,
        bottomMargin=38
    )

    DARK_BLUE = colors.HexColor('#1F4E79') # Exact reference header blue

    # Styles in Times-Roman
    title_style = ParagraphStyle(
        'RefTitle',
        fontName='Times-Bold',
        fontSize=15,
        leading=18,
        alignment=1, # Center
        textColor=colors.black,
        spaceAfter=10
    )

    meta_style = ParagraphStyle(
        'RefMeta',
        fontName='Times-Roman',
        fontSize=9,
        leading=12.5,
        alignment=0, # Left
        textColor=colors.black
    )

    sec_head_style = ParagraphStyle(
        'RefSecHead',
        fontName='Times-Bold',
        fontSize=9.5,
        leading=12.5,
        textColor=colors.black,
        spaceBefore=5,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'RefBody',
        fontName='Times-Roman',
        fontSize=8.2,
        leading=10.6,
        textColor=colors.black,
        alignment=4 # Justify
    )

    caption_style = ParagraphStyle(
        'RefCaption',
        fontName='Times-Italic',
        fontSize=7.2,
        leading=9,
        textColor=colors.black,
        alignment=1 # Center
    )

    tbl_hdr_style = ParagraphStyle(
        'RefTblHdr',
        fontName='Times-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=0 # Left
    )

    tbl_hdr_center = ParagraphStyle(
        'RefTblHdrCenter',
        fontName='Times-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1 # Center
    )

    tbl_cell_style = ParagraphStyle(
        'RefTblCell',
        fontName='Times-Roman',
        fontSize=7.2,
        leading=9,
        textColor=colors.black,
        alignment=0 # Left
    )

    tbl_cell_center = ParagraphStyle(
        'RefTblCellCenter',
        fontName='Times-Roman',
        fontSize=7.2,
        leading=9,
        textColor=colors.black,
        alignment=1 # Center
    )

    ref_style = ParagraphStyle(
        'RefBib',
        fontName='Times-Roman',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.black,
        alignment=4
    )

    story = []

    # =========================================================================
    # PAGE 1
    # =========================================================================
    
    # Title
    story.append(Paragraph("Kidney Abnormality Detection from CT (PS-21)", title_style))
    story.append(Spacer(1, 2))

    # Header Meta block
    meta_text = (
        "Name: Kanchamreddy Sai Rutwik Reddy | Register No.: RA2311026050138<br/>"
        "Department: IVCSE-AIML C | Date: 12 November 2026<br/>"
        "GitHub notebook: <font color='#1F4E79'><u>https://github.com/Rutwik-begin/PS-21-Kidney-Abnormality-Detection-from-CT-Domain-Healthcare-Medical-Imaging.git</u></font>"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 6))

    # 1. Problem and Objective
    story.append(Paragraph("1. Problem and Objective", sec_head_style))
    p1 = (
        "This task classifies kidney CT slices into four categories: normal, cyst, stone, or tumour. Early identification of renal abnormalities supports "
        "timely clinical triage and treatment planning. The CT Kidney dataset is imbalanced and includes multiple slices from the same patient study; "
        "therefore, splitting by patient ID is essential to avoid information leakage. Macro F1 is the primary metric because it weights each class equally, "
        "and stone recall is reported separately because stone is the smallest class."
    )
    story.append(Paragraph(p1, body_style))
    story.append(Spacer(1, 4))

    # 2. Dataset and Pre-processing
    story.append(Paragraph("2. Dataset and Pre-processing", sec_head_style))
    p2 = (
        "The CT Kidney Dataset contains 12,446 images: normal 5,077; cyst 3,709; tumour 2,283; stone 1,377. Source: Kaggle, "
        "https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone. Proposed split: approximately 70/15/15% by patient study ID, "
        "stratified as closely as feasible; exact partition counts must be generated from metadata. Resize to 224&times;224 and apply ImageNet normalization. "
        "Training-only augmentation may include modest rotation, crop, colour jitter, and horizontal flip for bilateral anatomical preservation. "
        "Use class-weighted cross-entropy or focal loss to handle stone class imbalance. Validation and test images receive deterministic preprocessing only."
    )
    story.append(Paragraph(p2, body_style))
    story.append(Spacer(1, 2))

    # Fig. 1 Class counts chart
    fig1_path = "D:/CBL/figures/sample_fig1_class_dist.png"
    if os.path.exists(fig1_path):
        story.append(Image(fig1_path, width=380, height=128))
        story.append(Paragraph("Fig. 1. Class counts in the full dataset; source counts, not model results.", caption_style))
    story.append(Spacer(1, 4))

    # 3. Methodology
    story.append(Paragraph("3. Methodology", sec_head_style))
    p3 = (
        "A pretrained EfficientNet-B0 is proposed with dropout (0.3) and a four-class output head. First train the head with the backbone frozen, then fine-tune "
        "the full network at a lower learning rate. Use AdamW, weighted focal loss / cross-entropy, cosine learning-rate decay and early stopping based on "
        "validation macro F1. Compare against ResNet-50 using the same data split and preprocessing. Fix random seeds and save split assignments and the best "
        "checkpoint for reproducibility."
    )
    story.append(Paragraph(p3, body_style))
    story.append(Spacer(1, 3))

    # Methodology Table
    tbl_method_data = [
        [
            Paragraph("Setting", tbl_hdr_style),
            Paragraph("Illustrative value", tbl_hdr_style),
            Paragraph("Setting", tbl_hdr_style),
            Paragraph("Illustrative value", tbl_hdr_style)
        ],
        [
            Paragraph("Architecture", tbl_cell_style),
            Paragraph("EfficientNet-B0 pretrained", tbl_cell_style),
            Paragraph("Optimizer", tbl_cell_style),
            Paragraph("AdamW; weight decay 1e-4", tbl_cell_style)
        ],
        [
            Paragraph("Input size", tbl_cell_style),
            Paragraph("224 &times; 224", tbl_cell_style),
            Paragraph("Learning rate", tbl_cell_style),
            Paragraph("1e-3 head; 1e-4 fine-tune", tbl_cell_style)
        ],
        [
            Paragraph("Loss", tbl_cell_style),
            Paragraph("Class-weighted Focal Loss / CE", tbl_cell_style),
            Paragraph("Batch size", tbl_cell_style),
            Paragraph("32", tbl_cell_style)
        ],
        [
            Paragraph("Epochs", tbl_cell_style),
            Paragraph("30 maximum", tbl_cell_style),
            Paragraph("Early stopping", tbl_cell_style),
            Paragraph("Patience 5; val. macro F1", tbl_cell_style)
        ]
    ]
    t_method = Table(tbl_method_data, colWidths=[110, 150, 100, 164])
    t_method.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#D3D3D3')),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
    ]))
    story.append(t_method)
    story.append(Spacer(1, 4))

    # 4. Results
    story.append(Paragraph("4. Results", sec_head_style))
    tbl_res_data = [
        [
            Paragraph("Model (synthetic example)", tbl_hdr_style),
            Paragraph("Test accuracy", tbl_hdr_center),
            Paragraph("Macro F1", tbl_hdr_center),
            Paragraph("Stone Recall", tbl_hdr_center),
            Paragraph("Normal F1", tbl_hdr_center),
            Paragraph("Train time*", tbl_hdr_center)
        ],
        [
            Paragraph("ResNet-50 baseline", tbl_cell_style),
            Paragraph("94.20%", tbl_cell_center),
            Paragraph("0.938", tbl_cell_center),
            Paragraph("0.782", tbl_cell_center),
            Paragraph("0.966", tbl_cell_center),
            Paragraph("18 min", tbl_cell_center)
        ],
        [
            Paragraph("EfficientNet-B0", tbl_cell_style),
            Paragraph("96.60%", tbl_cell_center),
            Paragraph("0.962", tbl_cell_center),
            Paragraph("0.948", tbl_cell_center),
            Paragraph("0.978", tbl_cell_center),
            Paragraph("14 min", tbl_cell_center)
        ]
    ]
    t_res = Table(tbl_res_data, colWidths=[164, 72, 72, 72, 72, 72])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#D3D3D3')),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 2))
    
    note_results = (
        "<font size=6.8 color='#333333'>Synthetic EfficientNet example (n=1,867): accuracy 96.60%; macro precision 0.964; macro recall 0.960; macro F1 0.962; "
        "stone recall 0.948. Training times and all model results are invented for layout demonstration; replace with actual notebook outputs.</font>"
    )
    story.append(Paragraph(note_results, body_style))
    story.append(Spacer(1, 3))

    # 5. Analysis and Discussion
    story.append(Paragraph("5. Analysis and Discussion", sec_head_style))
    p5 = (
        "In the illustrative matrix, normal is strongest because it has the most examples. Cyst and tumour are occasionally confused with each other, "
        "reflecting visual overlap in soft-tissue attenuation and necrotic regions. The rare class (stone) has the fewest training examples and small pixel footprints, "
        "so its recall is unstable under unweighted cross-entropy. The synthetic comparison suggests EfficientNet-B0 improves macro F1 and stone recall over the baseline; "
        "this is not evidence of a real gain. Random splitting across slices artificially inflates test performance due to patient leakage; for the actual run, "
        "enforce strict patient-study splitting and inspect misclassified images and train/validation curves before concluding whether overfitting occurred."
    )
    story.append(Paragraph(p5, body_style))

    # =========================================================================
    # PAGE 2
    # =========================================================================
    story.append(PageBreak())

    # Top title on Page 2 matching reference:
    story.append(Paragraph("Illustrative evaluation figures and deployment considerations", ParagraphStyle(
        'P2Title', fontName='Times-Bold', fontSize=10, leading=13, spaceAfter=4
    )))

    # Fig 2: Combined learning curves & confusion matrix
    fig2_path = "D:/CBL/figures/sample_fig2_combined.png"
    if os.path.exists(fig2_path):
        story.append(Image(fig2_path, width=440, height=155))
        story.append(Paragraph("Fig. 2. Synthetic training/validation curves (left) and synthetic test confusion matrix (right).", caption_style))
    story.append(Spacer(1, 6))

    # Per-class metrics table
    story.append(Paragraph("Illustrative per-class metrics", ParagraphStyle(
        'PerClassHead', fontName='Times-Bold', fontSize=9, leading=11, spaceAfter=2
    )))

    tbl_perclass_data = [
        [
            Paragraph("Class", tbl_hdr_style),
            Paragraph("Precision", tbl_hdr_style),
            Paragraph("Recall", tbl_hdr_style),
            Paragraph("F1", tbl_hdr_style),
            Paragraph("Support", tbl_hdr_style)
        ],
        [Paragraph("cyst", tbl_cell_style), Paragraph("0.96", tbl_cell_style), Paragraph("0.96", tbl_cell_style), Paragraph("0.96", tbl_cell_style), Paragraph("556", tbl_cell_style)],
        [Paragraph("normal", tbl_cell_style), Paragraph("0.98", tbl_cell_style), Paragraph("0.97", tbl_cell_style), Paragraph("0.98", tbl_cell_style), Paragraph("762", tbl_cell_style)],
        [Paragraph("stone", tbl_cell_style), Paragraph("0.95", tbl_cell_style), Paragraph("0.95", tbl_cell_style), Paragraph("0.95", tbl_cell_style), Paragraph("207", tbl_cell_style)],
        [Paragraph("tumor", tbl_cell_style), Paragraph("0.95", tbl_cell_style), Paragraph("0.92", tbl_cell_style), Paragraph("0.94", tbl_cell_style), Paragraph("342", tbl_cell_style)],
    ]
    t_pc = Table(tbl_perclass_data, colWidths=[100, 105, 105, 105, 109])
    t_pc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#D3D3D3')),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
    ]))
    story.append(t_pc)
    story.append(Spacer(1, 6))

    # 6. Real-world Relevance, Limitations and Ethics
    story.append(Paragraph("6. Real-world Relevance, Limitations and Ethics", sec_head_style))
    p6 = (
        "A validated classifier could help prioritize suspicious lesions (e.g. stones causing acute renal colic or renal cell carcinomas) for clinician review "
        "in emergency care or triage, but it must not replace medical diagnosis. The dataset may underrepresent rare histological variants, diverse scanner models, "
        "and pediatric protocols; external validation, calibration and prospective evaluation are necessary. False-negative stone or tumor predictions are especially "
        "serious. Medical images require appropriate permissions, secure storage and responsible handling."
    )
    story.append(Paragraph(p6, body_style))
    story.append(Spacer(1, 4))

    # 7. Conclusion and Future Work
    story.append(Paragraph("7. Conclusion and Future Work", sec_head_style))
    p7 = (
        "In this synthetic demonstration, EfficientNet-B0 has macro F1 0.962 versus 0.938 for ResNet-50, with stone recall of 0.948; these are not measured results. "
        "Next, run a reproducible patient-level experiment and replace all simulated metrics and curves with actual outputs, then validate on an external dataset."
    )
    story.append(Paragraph(p7, body_style))
    story.append(Spacer(1, 6))

    # References
    story.append(Paragraph("References", sec_head_style))
    r1 = (
        "[1] N. Islam et al., “CT Kidney Dataset: Normal-Cyst-Tumor and Stone,” Kaggle, 2021. "
        "<font color='#1F4E79'><u>https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone</u></font>."
    )
    r2 = (
        "[2] M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” in International Conference on Machine Learning (ICML), "
        "2019. <font color='#1F4E79'><u>https://arxiv.org/abs/1905.11946</u></font>."
    )
    story.append(Paragraph(r1, ref_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph(r2, ref_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] Successfully built exact reference replica PDF: {pdf_path}")

if __name__ == '__main__':
    build_pdf()

