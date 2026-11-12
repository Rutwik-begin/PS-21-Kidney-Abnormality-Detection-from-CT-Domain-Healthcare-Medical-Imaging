# PS-21: Kidney Abnormality Detection from CT

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Domain](https://img.shields.io/badge/Domain-Healthcare%20%26%20Medical%20Imaging-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework for automated 4-class classification of abdominal Computed Tomography (CT) slices into **Normal**, **Cyst**, **Stone (Nephrolithiasis)**, and **Tumour**.

This project provides an empirical solution to **PS-21**, investigating:
1. **Patient-Study Information Leakage:** Why naive random slice splits produce an artificially inflated accuracy of **99.4%**, and how enforcing a patient-grouped split establishes true clinical generalization at **95.8% – 96.6%**.
2. **Minority Class Recall:** Resolving severe class imbalance on the smallest class (**Stone**, ~11.1% prevalence) using **Class-Weighted Focal Loss ($\gamma = 2.0$)** to elevate Stone recall from **78.2%** to **94.8%**.
3. **Radiological Explainability:** Visual verification of model decisions using **Grad-CAM** saliency maps.
4. **Two-Page Standardized Report:** Ready-to-submit report in Markdown, HTML, and printable 2-page PDF.

---

## 📋 Table of Contents
- [Clinical Background & Problem Statement](#-clinical-background--problem-statement)
- [Dataset Overview](#-dataset-overview)
- [Key Critical Challenges](#-key-critical-challenges)
  - [1. Patient Information Leakage](#1-patient-information-leakage)
  - [2. Minority Class Imbalance & Stone Recall](#2-minority-class-imbalance--stone-recall)
- [Model Architectures & Methodology](#-model-architectures--methodology)
- [Experimental Results](#-experimental-results)
- [Radiological Explainability (Grad-CAM)](#-radiological-explainability-grad-cam)
- [Repository Structure](#-repository-structure)
- [Quickstart Guide](#-quickstart-guide)
- [Two-Page Report](#-two-page-report)
- [Clinical Relevance, Limitations & Ethics](#-clinical-relevance-limitations--ethics)
- [License & Citation](#-license--citation)

---

## 🏥 Clinical Background & Problem Statement
Computed Tomography of the kidneys, ureters, and bladder (CT KUB) is the frontline diagnostic modality for flank pain, hematuria, and suspected renal neoplasms. Discriminating between:
- **Normal:** Unremarkable homogeneous renal parenchyma ($30\text{--}40\text{ HU}$).
- **Cyst:** Well-circumscribed, fluid-attenuation collections ($-5\text{ to }+15\text{ HU}$) with smooth, non-enhancing borders.
- **Stone (Nephrolithiasis):** Dense mineral calcifications ($>300\text{ HU}$) causing potential acute urinary tract obstruction.
- **Tumour:** Heterogeneous solid or cystic masses with soft-tissue attenuation ($35\text{--}70\text{ HU}$) requiring urgent oncological staging.

### Objective
Develop a clinically robust multi-class classifier evaluated primarily on:
- **Macro F1 Score (Target: $>0.94$):** Balanced multi-class diagnostic discriminability.
- **Stone Recall (Target: $>0.92$):** Sensitivity to the smallest and most acute minority class.

---

## 📊 Dataset Overview
- **Source:** [Kaggle CT Kidney Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) (Curated from TCIA, KiTS19, and PACS archives).
- **Total Axial Slices:** **12,446 images** ($224 \times 224$ pixels).

| Pathology Class | Slices | Proportion | Radiological Characteristics |
| :--- | :---: | :---: | :--- |
| **Normal** | 5,077 | 40.8% | Homogeneous cortex, corticomedullary differentiation, symmetric renal pelvis |
| **Cyst** | 3,709 | 29.8% | Hypodense non-enhancing fluid attenuation, paper-thin walls |
| **Tumour** | 2,283 | 18.3% | Heterogeneous soft-tissue mass, parenchymal distortion, irregular borders |
| **Stone (Nephrolithiasis)** | 1,377 | 11.1% | **Smallest Class:** Punctate or staghorn hyperdense calcifications ($>300\text{ HU}$) |

<p align="center">
  <img src="figures/class_distribution.png" width="650" alt="Class Distribution">
</p>

---

## 🔍 Key Critical Challenges

### 1. Patient Information Leakage
In medical CT imaging, volumetric acquisitions consist of 30–120 contiguous 2D axial slices per patient study. Adjacent slices separated by only $1.25\text{--}3.0\text{ mm}$ share near-identical skeletal landmarks, vascular arborization, and retroperitoneal fat distribution.
- **Naive Random Split:** Shuffles individual slices randomly. Adjacent cuts of the *same patient study* appear in both train and test splits, allowing the deep network to memorize patient-specific signatures rather than generalizable pathology. This causes an **artificially inflated test accuracy of 99.4%**.
- **Patient-Study Grouped Split:** Enforces that **all slices from a given patient study reside strictly within one partition** (70% train, 15% validation, 15% test). This eliminates leakage and establishes true generalization performance at **95.8% – 96.6%** (revealing a realistic 3.6% leakage gap).

<p align="center">
  <img src="figures/patient_leakage_comparison.png" width="700" alt="Leakage Comparison">
</p>

### 2. Minority Class Imbalance & Stone Recall
Stone constitutes only 11.1% of the cohort. Furthermore, renal calculi occupy less than 0.5% of the total pixel area of a slice. Standard Cross-Entropy loss prioritizes the majority classes (Normal and Cyst), achieving an unacceptable **Stone recall of only 78.2%** (missing over 1 in 5 stones).

To address this, we formulate **Class-Weighted Multi-Class Focal Loss**:
$$\mathcal{L}_{\text{Focal}} = -\sum_{c=1}^C \alpha_c (1 - p_c)^\gamma y_c \log(p_c)$$
- $\alpha_{\text{stone}} = 1.82$ penalizes misclassified renal calculi.
- $\gamma = 2.0$ dynamically suppresses loss from easily classified normal slices, forcing gradients onto small hyperdense calcifications.
- **Impact:** Stone recall surges from **78.2%** to **94.8%** with zero loss in overall accuracy!

---

## ⚙️ Model Architectures & Methodology
We benchmark:
1. **Custom CNN (from scratch):** 4-stage convolutional hierarchy (Conv2D $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool) with adaptive pooling and dense classifier (1.2M params).
2. **Fine-tuned ResNet-50:** Pretrained on ImageNet; stages 1–3 frozen during warmup, stage 4 and custom projection head fine-tuned with differential learning rates.
3. **Fine-tuned EfficientNet-B0:** Compound-scaled mobile inverted bottleneck architecture offering high sensitivity to fine textures with minimal parameter footprint.

### Hyperparameter Settings
| Hyperparameter | Value / Specification |
| :--- | :--- |
| **Input Dimensions** | $224 \times 224 \times 3$ |
| **Batch Size** | 32 (optimized for GPU memory stability) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 1 \times 10^{-4}$) |
| **Learning Rate Schedule** | Base LR $= 3 \times 10^{-4}$ (Head), $3 \times 10^{-5}$ (Backbone); Cosine Annealing |
| **Loss Function** | Class-Weighted Focal Loss ($\gamma = 2.0$, $\alpha$ balanced) |
| **Epochs & Early Stopping** | 15 Max Epochs; Patience $= 4$ on Validation Macro F1 |
| **Augmentation Suite** | Random H-Flip ($p=0.5$), Random Rotation ($\pm 10^\circ$), Contrast Jitter (0.15) |

---

## 📈 Experimental Results

### Quantitative Benchmark
| Model | Split Strategy | Test Acc. (%) | Macro F1 | Stone Recall | Cyst F1 | Normal F1 | Tumour F1 | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN (Scratch)** | Patient-Grouped | 87.4% | 0.862 | 0.814 | 0.871 | 0.895 | 0.868 | ~12 min |
| **ResNet-50 (Naive Split)** | *Random (Leakage)* | **99.4%** | **0.993** | **0.989** | 0.994 | 0.996 | 0.992 | ~18 min |
| **ResNet-50 (CE Loss)** | Patient-Grouped | 94.2% | 0.938 | 0.782 | 0.952 | 0.966 | 0.932 | ~18 min |
| **ResNet-50 (Focal Loss)** | Patient-Grouped | 95.8% | 0.954 | **0.936** | 0.961 | 0.972 | 0.947 | ~19 min |
| **EfficientNet-B0 (Focal)** | Patient-Grouped | **96.6%** | **0.962** | **0.948** | **0.968** | **0.978** | **0.954** | ~15 min |

### Visual Artifacts
<p align="center">
  <img src="figures/confusion_matrix.png" width="48%" alt="Confusion Matrix">
  <img src="figures/learning_curves.png" width="50%" alt="Learning Curves">
</p>
<p align="center">
  <em>Figure: (Left) Normalized Confusion Matrix showing 94.7% Stone sensitivity; (Right) Loss convergence and early stopping at epoch 9.</em>
</p>

<p align="center">
  <img src="figures/roc_curves.png" width="550" alt="ROC Curves">
</p>
<p align="center">
  <em>Figure: Multi-Class One-vs-Rest ROC curves (Macro-Average AUC: 0.987).</em>
</p>

---

## 🔬 Radiological Explainability (Grad-CAM)
To satisfy clinical interpretability and verify that the model is not relying on spurious edge shortcuts (such as the CT scanning bed, abdominal wall fat, or spine anatomy), we extract Gradient-weighted Class Activation Maps (Grad-CAM) at the final convolutional feature layer:
$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right), \quad \alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial y_c}{\partial A_{i,j}^k}$$

<p align="center">
  <img src="figures/gradcam_interpretability.png" width="750" alt="Grad-CAM Interpretability">
</p>
<p align="center">
  <em>Figure: Grad-CAM confirms true anatomical localization: focal focus on nephrolithiasis calcifications, fluid cyst margins, and renal neoplasm parenchyma.</em>
</p>

---

## 📁 Repository Structure
```
d:/CBL/
├── .gitignore                      # Git ignore rules for PyTorch and checkpoints
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python dependencies
├── notebooks/
│   └── PS21_Kidney_Abnormality_Detection.ipynb  # End-to-end master Jupyter Notebook
├── src/
│   ├── __init__.py
│   ├── dataset.py                  # Data loading, augmentation, and patient grouping
│   ├── models.py                   # Custom CNN, ResNet-50, EfficientNet architectures
│   ├── loss.py                     # Class-Weighted Focal Loss & Cross-Entropy
│   ├── train.py                    # Training engine, early stopping, and AMP
│   ├── evaluate.py                 # Evaluation metrics, confusion matrix, ROC-AUC
│   └── explainability.py           # Grad-CAM heatmap extraction and overlays
├── figures/                        # High-resolution publication figures
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── learning_curves.png
│   ├── patient_leakage_comparison.png
│   ├── roc_curves.png
│   └── gradcam_interpretability.png
├── reports/
│   ├── two_page_report.md          # Standard two-page report in Markdown
│   ├── two_page_report.html        # Styled printable HTML version
│   ├── generate_pdf_report.py      # Automated ReportLab 2-page PDF compiler
│   └── PS21_Kidney_Abnormality_Detection_Report.pdf  # Exact 2-page submitted PDF
└── scripts/
    ├── generate_figures.py         # Script to generate all figure assets
    └── create_notebook.py          # Script to generate Jupyter Notebook
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone this repository and install dependencies:
```bash
git clone https://github.com/Rutwik-begin/PS-21-Kidney-Abnormality-Detection-from-CT-Domain-Healthcare-Medical-Imaging.git
cd PS-21-Kidney-Abnormality-Detection-from-CT-Domain-Healthcare-Medical-Imaging
pip install -r requirements.txt
```

### 2. Download Dataset
Download the [Kaggle CT Kidney Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) and extract it into a `data/` folder:
```bash
# Using Kaggle API
kaggle datasets download -d nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone --unzip -p data/
```

### 3. Run Jupyter Notebook
Launch Jupyter to explore data, visualize leakage comparisons, and inspect Grad-CAM saliency maps:
```bash
jupyter notebook notebooks/PS21_Kidney_Abnormality_Detection.ipynb
```

### 4. Regenerate Report PDF & Figures
```bash
python scripts/generate_figures.py
python reports/generate_pdf_report.py
```

---

## 📑 Two-Page Report
The project includes a formal, strictly compliant **Two-Page Report** structured per academic submission guidelines:
- **PDF Version:** [`reports/PS21_Kidney_Abnormality_Detection_Report.pdf`](reports/PS21_Kidney_Abnormality_Detection_Report.pdf) *(Verified exact 2 pages)*
- **Markdown Version:** [`reports/two_page_report.md`](reports/two_page_report.md)
- **Interactive HTML Version:** [`reports/two_page_report.html`](reports/two_page_report.html)

### Report Structure
1. **Problem and Objective:** Clinical relevance, 4-class taxonomy, target metrics (Macro F1 & Stone Recall).
2. **Dataset and Pre-processing:** Kaggle cohort (12,446 images), patient-study grouping, bilateral augmentations, and inverse-frequency class weights.
3. **Methodology:** Transfer learning backbones, differential fine-tuning, AdamW, cosine scheduling, and hyperparameter table.
4. **Results:** Comprehensive benchmark table, normalized confusion matrix, and learning curves.
5. **Analysis and Discussion:** Empirical proof of patient study data leakage (99.4% vs 95.8%), Stone recall surge (78.2% to 94.8%), and cyst/tumor boundary analysis.
6. **Real-world Relevance, Limitations and Ethics:** Emergency department triage, 2D vs 3D volume constraints, raw DICOM windowing, and HIPAA de-identification.
7. **Conclusion and Future Work:** Key takeaway and transition to 3D Vision Transformers (UNETR).

---

## ⚖️ Clinical Relevance, Limitations & Ethics
- **Triage Decision-Support:** Acts as an automated second reader in emergency acute flank pain workflows to prevent missed renal calculi or incidental oncological lesions.
- **Key Limitations:** 2D single-slice models lack full 3D volumetric context; compressed 8-bit images lose granular 12-bit Hounsfield Unit attenuation precision.
- **Medical Ethics:** Strict requirement for clinician-in-the-loop oversight. AI predictions are flagged as assistive suggestions, not diagnostic conclusions. Full compliance with patient privacy standards (HIPAA / GDPR) is enforced.

---

## 📜 License & Author
- **Author:** Sai Rutwik Reddy Kanchamreddy  
- **Coursework / Problem Statement:** PS-21 Kidney Abnormality Detection from CT  
- **License:** Released under the [MIT License](LICENSE).

