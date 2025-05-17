from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, PretrainedConfig
from datasets import Dataset
from training.collator import collate_fn
from .callbacks import LossCallback
from .metrics import compute_metrics
from model.base_model import load_base_model
from model.classifier import BiDoRAClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch

def run_cross_validation(balanced_df, feature_columns):
    models = []
    X_all = balanced_df[feature_columns].values
    y_all = balanced_df['labels'].values
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    input_size = X_all.shape[1]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"Fold {fold+1}")
        train_df = balanced_df.iloc[train_idx]
        val_df = balanced_df.iloc[val_idx]

        ds_train = Dataset.from_pandas(train_df).map(lambda ex: {'x': [ex[c] for c in feature_columns], 'labels': ex['labels']})
        ds_val = Dataset.from_pandas(val_df).map(lambda ex: {'x': [ex[c] for c in feature_columns], 'labels': ex['labels']})

        base = load_base_model(input_size, 512, "checkpoints/models.ckpt")
        model = BiDoRAClassifier(base, input_size, 2)

        with torch.no_grad():
            tmp = nn.Linear(256, 2)
            tmp.reset_parameters()
            model.head.bidora.base_weight.copy_(tmp.weight)

        model.config = PretrainedConfig(model_type='custom')

        args = TrainingArguments(
            output_dir=f'./results/fold{fold}',
            num_train_epochs=30,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=8,
            learning_rate=1e-4,
            warmup_steps=200,
            weight_decay=1e-4,
            lr_scheduler_type='cosine_with_restarts',
            logging_dir='./logs',
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='auc',
            dataloader_pin_memory=False,
            greater_is_better=True,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5), LossCallback()]
        )

        trainer.train()
        models.append(model)

    return models