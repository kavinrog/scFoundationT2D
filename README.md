# scFoundation-T2D: Single-Cell Transcriptomics Classifier for Type 2 Diabetes

This repository provides a deep learning pipeline to classify Type 2 Diabetes (T2D) status from single-cell transcriptomic data using a pretrained encoder and a custom BiDoRA classifier architecture. The pipeline integrates data preprocessing, MixUp regularization, focal loss, and ensemble evaluation to ensure robustness on imbalanced datasets.

---

## 🧬 Dataset

- **Gene Expression Matrix**: `data/diabetes_beta_scrna.csv`
- **Metadata**: `data/SraRunTable.csv`
- Assumes single-cell RNA-seq data and donor annotations.

---

## 🔧 Features

- ✅ Preprocessing: log normalization, z-scoring, and random oversampling.
- ✅ Pretrained Encoder (scFoundation-based).
- ✅ BiDoRA linear classifier with Focal Loss.
- ✅ MixUp augmentation during training.
- ✅ Stratified 5-fold cross-validation.
- ✅ Ensemble evaluation with AUC, F1, and plots.
- ✅ Device-compatible: CPU / CUDA / Apple MPS.

---

## 📁 Project Structure

scFoundation-T2D/
├── checkpoints/           # Place pretrained models.ckpt here (ignored by git)
├── data/
│   ├── diabetes_beta_scrna.csv
│   └── SraRunTable.csv
├── model/
│   ├── base_model.py
│   ├── bidora.py
│   ├── classifier.py
│   └── loss.py
├── training/
│   ├── train.py
│   ├── collator.py
│   ├── metrics.py
│   ├── callbacks.py
├── main.py                # Entry point for training & evaluation
├── requirements.txt       # Dependencies
└── README.md

---

## 📦 Setup

### 1. Create and activate virtual environment (optional but recommended)

```bash
python -m venv scft2d
source scft2d/bin/activate  # or scft2d\Scripts\activate on Windows

2. Install dependencies

pip install -r requirements.txt

3. Place the pretrained encoder

Download or locate the pretrained encoder file and place it at:

checkpoints/models.ckpt

This file is not included in the repo due to size.

⸻

🚀 Run the Pipeline

python main.py

This will:
	•	Load and preprocess the data
	•	Train across 5 folds
	•	Evaluate an ensemble on a hold-out test set
	•	Output AUC, F1, confusion matrix, and plots

⸻

📊 Output
	•	📈 ROC Curve
	•	🔲 Confusion Matrix
	•	📊 Bar Plot of AUC and F1 Score
	•	📋 Console logs of training + metrics

⸻

📝 Notes
	•	Designed for small-to-medium scale single-cell studies
	•	Ideal for classification of clinical phenotypes from RNA-seq
	•	For reproducibility, random seeds are fixed (e.g., random_state=42)

⸻

📮 Contact

For questions, feel free to open an issue or reach out directly.

⸻


---

Let me know if you want the Markdown saved into a file or included in your GitHub upload instructions.