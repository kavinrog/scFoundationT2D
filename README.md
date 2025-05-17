# scFoundation-T2D: Single-Cell Transcriptomics Classifier for Type 2 Diabetes

This repository provides a deep learning pipeline to classify Type 2 Diabetes (T2D) status from single-cell transcriptomic data using a pretrained encoder and a custom BiDoRA classifier architecture. The pipeline integrates data preprocessing, MixUp regularization, focal loss, and ensemble evaluation to ensure robustness on imbalanced datasets.

This work is part of an ongoing research project under the guidance of Prof. Pengtao Xie at the PXie Lab, University of California, San Diego.

## Dataset

* **Gene Expression Matrix**: `data/diabetes_beta_scrna.csv`
* **Metadata**: `data/SraRunTable.csv`

Make sure these files are placed inside the `data/` directory as shown in the project structure below. These are included in the provided `scFoundation-T2D-assets.zip` archive.

---

## Features

* ✅ Preprocessing: log normalization, z-scoring, and random oversampling.
* ✅ Pretrained Encoder (scFoundation-based).
* ✅ BiDoRA linear classifier with Focal Loss.
* ✅ MixUp augmentation during training.
* ✅ Stratified 5-fold cross-validation.
* ✅ Ensemble evaluation with AUC, F1, and plots.
* ✅ Device-compatible: CPU / CUDA / Apple MPS.

---

## Project Structure

Here's the folder layout expected after setup:

```
scFoundation-T2D/
├── checkpoints/
│   └── models.ckpt
├── data/
│   ├── diabetes_beta_scrna.csv
│   ├── SraRunTable.csv
│   ├── load_data.py
│   └── preprocess.py
├── model/
│   ├── base_model.py
│   ├── bidora.py
│   ├── classifier.py
│   └── loss.py
├── training/
│   ├── __init__.py
│   ├── callbacks.py
│   ├── collator.py
│   ├── metrics.py
│   └── train.py
├── .gitignore
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Cloning the Repository

```bash
git clone https://github.com/your-username/scFoundation-T2D.git
cd scFoundation-T2D
```

---

## Setup

### 1. Create and activate virtual environment (optional but recommended)

```bash
python -m venv scft2d
source scft2d/bin/activate  # or scft2d\Scripts\activate on Windows
```

### 2. Install or upgrade pip and dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the dataset and model checkpoint

Due to GitHub's file size limitations, the dataset and model checkpoint are not included in this repository.

👉 **Please contact the author directly to receive the following files:**

```
data/
├── diabetes_beta_scrna.csv
└── SraRunTable.csv

checkpoints/
└── models.ckpt
```

Once received, **manually place them** into the appropriate directories as shown above:

* Place `.csv` files inside the `data/` folder
* Place `models.ckpt` inside the `checkpoints/` folder

---

## Run the Pipeline

```bash
python main.py
```

This will:

* Load and preprocess the data
* Train across 5 folds
* Evaluate an ensemble on a hold-out test set
* Output AUC, F1, confusion matrix, and plots

---

## Output

* ROC Curve
* Confusion Matrix
* Bar Plot of AUC and F1 Score
* Console logs of training + metrics

---

## Notes

* Designed for small-to-medium scale single-cell studies
* Ideal for classification of clinical phenotypes from RNA-seq
* For reproducibility, random seeds are fixed (e.g., random\_state=42)

---

## 📬 Contact

For questions, feel free to open an issue or reach out directly.

---
