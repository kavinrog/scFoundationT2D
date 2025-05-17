import numpy as np

def preprocess(df, selected_columns):
    df = df.copy()
    gene_cols = [c for c in df.columns if c not in selected_columns]
    df[gene_cols] = np.log1p(df[gene_cols])
    df[gene_cols] = df[gene_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    return df.dropna().reset_index(drop=True), gene_cols