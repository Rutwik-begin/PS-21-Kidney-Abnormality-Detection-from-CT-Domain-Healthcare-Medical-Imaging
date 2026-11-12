"""
generate_docx_report.py - Creates the two-page report in Microsoft Word (.docx) format
replicating the exact visual design, tables, fonts, colors, and layout of the reference document.
"""

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=60, bottom=60, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_docx():
    doc = docx.Document()

    # Set page margins: 0.6 inch (43.2 pt)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.55)
        section.bottom_margin = Inches(0.55)
        section.left_margin = Inches(0.65)
        section.right_margin = Inches(0.65)

    # Base style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(8.5)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

    # ==================== PAGE 1 ====================
    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(10)
    run_title = p_title.add_run("Kidney Abnormality Detection from CT (PS-21)")
    run_title.font.name = 'Times New Roman'
    run_title.font.size = Pt(15)
    run_title.font.bold = True

    # Metadata
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_after = Pt(8)
    p_meta.paragraph_format.line_spacing = 1.2
    
    r1 = p_meta.add_run("Name: Kanchamreddy Sai Rutwik Reddy | Register No.: RA2311026050138\n")
    r2 = p_meta.add_run("Department: IVCSE-AIML C | Date: 12 November 2026\n")
    r3 = p_meta.add_run("GitHub notebook: ")
    r4 = p_meta.add_run("https://github.com/Rutwik-begin/PS-21-Kidney-Abnormality-Detection-from-CT-Domain-Healthcare-Medical-Imaging.git")
    r4.font.color.rgb = RGBColor(31, 78, 121)
    r4.font.underline = True

    # 1. Problem and Objective
    p_h1 = doc.add_paragraph()
    p_h1.paragraph_format.space_before = Pt(6)
    p_h1.paragraph_format.space_after = Pt(2)
    r_h1 = p_h1.add_run("1. Problem and Objective")
    r_h1.font.bold = True
    r_h1.font.size = Pt(9.5)

    p_p1 = doc.add_paragraph(
        "This task classifies kidney CT slices into four categories: normal, cyst, stone, or tumour. Early identification of renal abnormalities "
        "supports timely clinical triage and treatment planning. The CT Kidney dataset is imbalanced and includes multiple slices from the same "
        "patient study; therefore, splitting by patient ID is essential to avoid information leakage. Macro F1 is the primary metric because it "
        "weights each class equally, unlike accuracy, which can be dominated by common normal slices. Stone recall is reported separately because "
        "stone is the smallest class."
    )
    p_p1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # 2. Dataset and Pre-processing
    p_h2 = doc.add_paragraph()
    p_h2.paragraph_format.space_before = Pt(6)
    p_h2.paragraph_format.space_after = Pt(2)
    r_h2 = p_h2.add_run("2. Dataset and Pre-processing")
    r_h2.font.bold = True
    r_h2.font.size = Pt(9.5)

    p_p2 = doc.add_paragraph(
        "The CT Kidney Dataset contains 12,446 images: normal 5,077; cyst 3,709; tumour 2,283; stone 1,377. Source: Kaggle, "
        "https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone. Proposed split: approximately 70/15/15% by patient study ID, "
        "stratified as closely as feasible; exact partition counts must be generated from metadata. Resize to 224×224 and apply ImageNet normalization. "
        "Training-only augmentation may include modest rotation, crop and colour jitter; validate horizontal flip for bilateral kidney preservation. "
        "Use class-weighted cross-entropy or focal loss if class imbalance remains. Validation and test images receive deterministic preprocessing only."
    )
    p_p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Fig 1 Image
    fig1_path = "D:/CBL/figures/sample_fig1_class_dist.png"
    if os.path.exists(fig1_path):
        p_fig1 = doc.add_paragraph()
        p_fig1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig1.paragraph_format.space_before = Pt(2)
        p_fig1.paragraph_format.space_after = Pt(1)
        p_fig1.add_run().add_picture(fig1_path, width=Inches(4.8))

        p_cap1 = doc.add_paragraph()
        p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap1.paragraph_format.space_after = Pt(6)
        r_cap1 = p_cap1.add_run("Fig. 1. Class counts in the full dataset; source counts, not model results.")
        r_cap1.font.italic = True
        r_cap1.font.size = Pt(7.5)

    # 3. Methodology
    p_h3 = doc.add_paragraph()
    p_h3.paragraph_format.space_before = Pt(4)
    p_h3.paragraph_format.space_after = Pt(2)
    r_h3 = p_h3.add_run("3. Methodology")
    r_h3.font.bold = True
    r_h3.font.size = Pt(9.5)

    p_p3 = doc.add_paragraph(
        "A pretrained EfficientNet-B0 is proposed with dropout (0.3) and a four-class output head. First train the head with the backbone frozen, then "
        "fine-tune the full network at a lower learning rate. Use AdamW, weighted focal loss / cross-entropy, cosine learning-rate decay and early "
        "stopping based on validation macro F1. Compare against ResNet-50 using the same data split and preprocessing. Fix random seeds and save split "
        "assignments and the best checkpoint for reproducibility."
    )
    p_p3.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Methodology Table
    t_m = doc.add_table(rows=5, cols=4)
    t_m.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_m)

    headers_m = ["Setting", "Illustrative value", "Setting", "Illustrative value"]
    for i, h in enumerate(headers_m):
        cell = t_m.cell(0, i)
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=40, bottom=40, left=80, right=80)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(7.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    rows_m = [
        ("Architecture", "EfficientNet-B0 pretrained", "Optimizer", "AdamW; weight decay 1e-4"),
        ("Input size", "224 × 224", "Learning rate", "1e-3 head; 1e-4 fine-tune"),
        ("Loss", "Class-weighted Focal Loss / CE", "Batch size", "32"),
        ("Epochs", "30 maximum", "Early stopping", "Patience 5; val. macro F1")
    ]
    for r_idx, row_vals in enumerate(rows_m, start=1):
        for c_idx, val in enumerate(row_vals):
            cell = t_m.cell(r_idx, c_idx)
            set_cell_margins(cell, top=30, bottom=30, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.size = Pt(7.5)

    # 4. Results
    p_h4 = doc.add_paragraph()
    p_h4.paragraph_format.space_before = Pt(6)
    p_h4.paragraph_format.space_after = Pt(2)
    r_h4 = p_h4.add_run("4. Results")
    r_h4.font.bold = True
    r_h4.font.size = Pt(9.5)

    t_r = doc.add_table(rows=3, cols=6)
    t_r.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_r)

    headers_r = ["Model (synthetic example)", "Test accuracy", "Macro F1", "Stone Recall", "Normal F1", "Train time*"]
    for i, h in enumerate(headers_r):
        cell = t_r.cell(0, i)
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=40, bottom=40, left=60, right=60)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        if i > 0:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(7.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    rows_r = [
        ("ResNet-50 baseline", "94.20%", "0.938", "0.782", "0.966", "18 min"),
        ("EfficientNet-B0", "96.60%", "0.962", "0.948", "0.978", "14 min")
    ]
    for r_idx, row_vals in enumerate(rows_r, start=1):
        for c_idx, val in enumerate(row_vals):
            cell = t_r.cell(r_idx, c_idx)
            set_cell_margins(cell, top=30, bottom=30, left=60, right=60)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if c_idx > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(7.5)

    p_note = doc.add_paragraph()
    p_note.paragraph_format.space_before = Pt(2)
    p_note.paragraph_format.space_after = Pt(4)
    r_note = p_note.add_run(
        "Synthetic EfficientNet example (n=1,867): accuracy 96.60%; macro precision 0.964; macro recall 0.960; macro F1 0.962; stone recall 0.948. "
        "Training times and all model results are invented for layout demonstration; replace with actual notebook outputs."
    )
    r_note.font.size = Pt(6.8)
    r_note.font.color.rgb = RGBColor(60, 60, 60)

    # 5. Analysis and Discussion
    p_h5 = doc.add_paragraph()
    p_h5.paragraph_format.space_before = Pt(4)
    p_h5.paragraph_format.space_after = Pt(2)
    r_h5 = p_h5.add_run("5. Analysis and Discussion")
    r_h5.font.bold = True
    r_h5.font.size = Pt(9.5)

    p_p5 = doc.add_paragraph(
        "In the illustrative matrix, normal is strongest because it has the most examples. Cyst and tumour are occasionally confused with each other, "
        "reflecting visual overlap in soft-tissue attenuation and necrotic regions. The rare class (stone) has the fewest training examples and small pixel footprints, "
        "so its recall is unstable under unweighted cross-entropy. The synthetic comparison suggests EfficientNet-B0 improves macro F1 and stone recall over the baseline; "
        "this is not evidence of a real gain. Random splitting across slices artificially inflates test performance due to patient leakage; for the actual run, "
        "enforce strict patient-study splitting and inspect misclassified images and train/validation curves before concluding whether overfitting occurred."
    )
    p_p5.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # ==================== PAGE BREAK ====================
    doc.add_page_break()

    # ==================== PAGE 2 ====================
    p_top2 = doc.add_paragraph()
    p_top2.paragraph_format.space_before = Pt(0)
    p_top2.paragraph_format.space_after = Pt(4)
    r_top2 = p_top2.add_run("Illustrative evaluation figures and deployment considerations")
    r_top2.font.bold = True
    r_top2.font.size = Pt(9.5)

    # Fig 2 Composite
    fig2_path = "D:/CBL/figures/sample_fig2_combined.png"
    if os.path.exists(fig2_path):
        p_fig2 = doc.add_paragraph()
        p_fig2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig2.paragraph_format.space_before = Pt(2)
        p_fig2.paragraph_format.space_after = Pt(1)
        p_fig2.add_run().add_picture(fig2_path, width=Inches(5.4))

        p_cap2 = doc.add_paragraph()
        p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap2.paragraph_format.space_after = Pt(8)
        r_cap2 = p_cap2.add_run("Fig. 2. Synthetic training/validation curves (left) and synthetic test confusion matrix (right).")
        r_cap2.font.italic = True
        r_cap2.font.size = Pt(7.5)

    # Per-class metrics title
    p_h_pc = doc.add_paragraph()
    p_h_pc.paragraph_format.space_before = Pt(4)
    p_h_pc.paragraph_format.space_after = Pt(2)
    r_h_pc = p_h_pc.add_run("Illustrative per-class metrics")
    r_h_pc.font.bold = True
    r_h_pc.font.size = Pt(9.0)

    # Per-class metrics Table
    t_pc = doc.add_table(rows=5, cols=5)
    t_pc.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_pc)

    headers_pc = ["Class", "Precision", "Recall", "F1", "Support"]
    for i, h in enumerate(headers_pc):
        cell = t_pc.cell(0, i)
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=40, bottom=40, left=80, right=80)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(7.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    rows_pc = [
        ("cyst", "0.96", "0.96", "0.96", "556"),
        ("normal", "0.98", "0.97", "0.98", "762"),
        ("stone", "0.95", "0.95", "0.95", "207"),
        ("tumor", "0.95", "0.92", "0.94", "342")
    ]
    for r_idx, row_vals in enumerate(rows_pc, start=1):
        for c_idx, val in enumerate(row_vals):
            cell = t_pc.cell(r_idx, c_idx)
            set_cell_margins(cell, top=30, bottom=30, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.size = Pt(7.5)

    # 6. Real-world Relevance, Limitations and Ethics
    p_h6 = doc.add_paragraph()
    p_h6.paragraph_format.space_before = Pt(8)
    p_h6.paragraph_format.space_after = Pt(2)
    r_h6 = p_h6.add_run("6. Real-world Relevance, Limitations and Ethics")
    r_h6.font.bold = True
    r_h6.font.size = Pt(9.5)

    p_p6 = doc.add_paragraph(
        "A validated classifier could help prioritize suspicious lesions (e.g. stones causing acute renal colic or renal cell carcinomas) for clinician review "
        "in emergency care or triage, but it must not replace medical diagnosis. The dataset may underrepresent rare histological variants, diverse scanner models, "
        "and pediatric protocols; external validation, calibration and prospective evaluation are necessary. False-negative stone or tumor predictions are especially "
        "serious. Medical images require appropriate permissions, secure storage and responsible handling."
    )
    p_p6.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # 7. Conclusion and Future Work
    p_h7 = doc.add_paragraph()
    p_h7.paragraph_format.space_before = Pt(6)
    p_h7.paragraph_format.space_after = Pt(2)
    r_h7 = p_h7.add_run("7. Conclusion and Future Work")
    r_h7.font.bold = True
    r_h7.font.size = Pt(9.5)

    p_p7 = doc.add_paragraph(
        "In this synthetic demonstration, EfficientNet-B0 has macro F1 0.962 versus 0.938 for ResNet-50, with stone recall of 0.948; these are not measured results. "
        "Next, run a reproducible patient-level experiment and replace all simulated metrics and curves with actual outputs, then validate on an external dataset."
    )
    p_p7.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # References
    p_href = doc.add_paragraph()
    p_href.paragraph_format.space_before = Pt(8)
    p_href.paragraph_format.space_after = Pt(2)
    r_href = p_href.add_run("References")
    r_href.font.bold = True
    r_href.font.size = Pt(9.0)

    p_r1 = doc.add_paragraph()
    p_r1.paragraph_format.space_after = Pt(2)
    r_r1_a = p_r1.add_run("[1] N. Islam et al., “CT Kidney Dataset: Normal-Cyst-Tumor and Stone,” Kaggle, 2021. ")
    r_r1_b = p_r1.add_run("https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone")
    r_r1_b.font.color.rgb = RGBColor(31, 78, 121)
    r_r1_b.font.underline = True

    p_r2 = doc.add_paragraph()
    p_r2.paragraph_format.space_after = Pt(2)
    r_r2_a = p_r2.add_run("[2] M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” in International Conference on Machine Learning (ICML), 2019. ")
    r_r2_b = p_r2.add_run("https://arxiv.org/abs/1905.11946")
    r_r2_b.font.color.rgb = RGBColor(31, 78, 121)
    r_r2_b.font.underline = True

    out_docx_path = Path("D:/CBL/reports/PS21_Kidney_Abnormality_Detection_Report.docx")
    doc.save(str(out_docx_path))
    print(f"[+] Successfully generated DOCX report: {out_docx_path}")

if __name__ == '__main__':
    build_docx()
