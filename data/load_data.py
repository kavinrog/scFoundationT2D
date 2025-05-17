import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from .preprocess import preprocess

def load_and_preprocess_data():
    data_df = pd.read_csv("data/diabetes_beta_scrna.csv")
    metadata_df = pd.read_csv("data/SraRunTable.csv")
    selected_columns = ['AGE','cell_subtype','condition','gender','tissue','ETHNICITY','donor_id']
    merged = pd.concat([data_df, metadata_df[selected_columns]], axis=1)

    processed_df, feature_columns = preprocess(merged, selected_columns)
    labels = processed_df['condition'].map({'non-diabetic': 0, 'T2D': 1})
    processed_df['labels'] = labels

    ros = RandomOverSampler(random_state=42)
    X_bal, y_bal = ros.fit_resample(processed_df[feature_columns], processed_df['labels'])

    balanced_df = pd.concat([pd.DataFrame(X_bal, columns=feature_columns), pd.DataFrame({'labels': y_bal})], axis=1)
    return balanced_df, feature_columns