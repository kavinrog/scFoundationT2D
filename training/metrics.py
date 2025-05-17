import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from datasets import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    auc = roc_auc_score(labels, probs[:, 1])
    f1 = f1_score(labels, preds, average='weighted')
    return {'f1_score': f1, 'auc': auc}

def evaluate_ensemble(models, balanced_df, feature_columns):
    # Set device (MPS > CUDA > CPU)
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Evaluating ensemble on device: {device}")

    # Prepare test data
    full_ds = Dataset.from_pandas(balanced_df)
    full_ds = full_ds.map(lambda ex: {'x': [ex[c] for c in feature_columns], 'labels': ex['labels']})
    full_ds = full_ds.remove_columns([c for c in balanced_df.columns if c not in ['x', 'labels']])
    test_split = full_ds.train_test_split(test_size=0.2, seed=42)
    test_ds = test_split['test']

    X_test = np.stack(test_ds['x'])
    y_test = np.array(test_ds['labels'])

    # Run ensemble prediction
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    probs = []

    for m in models:
        m.to(device)
        m.eval()
        with torch.no_grad():
            logits = m(X_tensor)['logits']
            softmax_probs = F.softmax(logits, dim=-1).cpu().numpy()
            probs.append(softmax_probs)

    mean_probs = np.mean(probs, axis=0)
    preds = np.argmax(mean_probs, axis=1)

    # Metrics
    auc = roc_auc_score(y_test, mean_probs[:, 1])
    f1 = f1_score(y_test, preds, average='weighted')

    print(f"\nEnsemble Test AUC: {auc:.4f}")
    print(f" Ensemble Test F1: {f1:.4f}")
    print("\n Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\n Classification Report:\n", classification_report(y_test, preds, target_names=['non-diabetic', 'T2D']))

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, mean_probs[:, 1])
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title("ROC Curve")
    plt.show()

    # --- Confusion Matrix ---
    ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=['non-diabetic', 'T2D'], cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    # --- Metrics Bar Plot ---
    plt.figure(figsize=(6, 4))
    plt.bar(["AUC", "F1 Score"], [auc, f1], color=['skyblue', 'salmon'])
    plt.ylim(0, 1)
    plt.title("Ensemble Performance Metrics")
    plt.ylabel("Score")
    plt.show()