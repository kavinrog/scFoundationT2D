from config import *
from data.load_data import load_and_preprocess_data
from training.train import run_cross_validation
from training.metrics import evaluate_ensemble
import torch

if __name__ == "__main__":
    balanced_df, feature_columns = load_and_preprocess_data()
    models = run_cross_validation(balanced_df, feature_columns)
    evaluate_ensemble(models, balanced_df, feature_columns)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")