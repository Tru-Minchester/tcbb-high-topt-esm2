from pathlib import Path
from typing import Dict
import warnings
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

TRAIN_CSV = BASE_DIR / "15_strict_nomixed_esm2_features.csv"
EXTERNAL_CSV = BASE_DIR / "24_external_esm2_features.csv"

RESULT_OUT = BASE_DIR / "25_external_test_results.csv"
SUMMARY_OUT = BASE_DIR / "25_external_test_summary.txt"

HASH_COL = "Sequence_Hash"
LABEL_COL = "Binary_Label"
TEMP_COL = "Target_Temperature"

N_SPLITS_INNER = 5
RANDOM_STATE = 42


def compute_metrics(y_true, y_prob, y_pred) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def build_pipeline():
    pipe = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
        "clf__class_weight": [None, "balanced_subsample"],
    }

    return pipe, param_grid


def main():
    print("=" * 90)
    print("Step 25: train final ESM2+RF model and evaluate on external novel40 set")
    print("=" * 90)
    print(f"[INFO] Train CSV: {TRAIN_CSV}")
    print(f"[INFO] External CSV: {EXTERNAL_CSV}")

    for p in [TRAIN_CSV, EXTERNAL_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    train_df = pd.read_csv(TRAIN_CSV)
    ext_df = pd.read_csv(EXTERNAL_CSV)

    esm_cols_train = [c for c in train_df.columns if c.startswith("ESM_")]
    esm_cols_ext = [c for c in ext_df.columns if c.startswith("ESM_")]

    common_esm_cols = sorted(set(esm_cols_train) & set(esm_cols_ext))
    if len(common_esm_cols) == 0:
        raise ValueError("No shared ESM feature columns found between train and external sets")

    required_train = [HASH_COL, LABEL_COL] + common_esm_cols
    required_ext = [HASH_COL, LABEL_COL] + common_esm_cols

    train_df = train_df.dropna(subset=[HASH_COL, LABEL_COL]).drop_duplicates(subset=[HASH_COL]).copy()
    ext_df = ext_df.dropna(subset=[HASH_COL, LABEL_COL]).drop_duplicates(subset=[HASH_COL]).copy()

    X_train = train_df[common_esm_cols].values
    y_train = train_df[LABEL_COL].astype(int).values

    X_ext = ext_df[common_esm_cols].values
    y_ext = ext_df[LABEL_COL].astype(int).values

    print(f"[INFO] Train samples: {len(train_df)}")
    print(f"[INFO] External samples: {len(ext_df)}")
    print(f"[INFO] Shared ESM dimensions: {len(common_esm_cols)}")
    print(f"[INFO] Train positives: {int(np.sum(y_train))}")
    print(f"[INFO] Train negatives: {int(len(y_train) - np.sum(y_train))}")
    print(f"[INFO] External positives: {int(np.sum(y_ext))}")
    print(f"[INFO] External negatives: {int(len(y_ext) - np.sum(y_ext))}")

    pipe, param_grid = build_pipeline()
    inner_cv = StratifiedKFold(
        n_splits=N_SPLITS_INNER,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_prob = best_model.predict_proba(X_ext)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_ext, y_prob, y_pred)

    result_df = ext_df[[HASH_COL]].copy()
    result_df["y_true"] = y_ext
    result_df["y_prob"] = y_prob
    result_df["y_pred"] = y_pred
    result_df.to_csv(RESULT_OUT, index=False)

    summary_lines = [
        "Step 25 summary",
        f"Train samples: {len(train_df)}",
        f"External samples: {len(ext_df)}",
        f"Shared ESM dimensions: {len(common_esm_cols)}",
        f"Train positives: {int(np.sum(y_train))}",
        f"Train negatives: {int(len(y_train) - np.sum(y_train))}",
        f"External positives: {int(np.sum(y_ext))}",
        f"External negatives: {int(len(y_ext) - np.sum(y_ext))}",
        f"Best params: {json.dumps(grid.best_params_, ensure_ascii=False)}",
        f"ROC_AUC: {metrics['ROC_AUC']:.4f}",
        f"PR_AUC: {metrics['PR_AUC']:.4f}",
        f"MCC: {metrics['MCC']:.4f}",
        f"F1: {metrics['F1']:.4f}",
        f"Balanced_Accuracy: {metrics['Balanced_Accuracy']:.4f}",
        f"TN: {metrics['TN']}",
        f"FP: {metrics['FP']}",
        f"FN: {metrics['FN']}",
        f"TP: {metrics['TP']}",
        f"Saved per-sample results: {RESULT_OUT.name}",
    ]
    SUMMARY_OUT.write_text("\n".join(summary_lines), encoding="utf-8")

    print("-" * 90)
    print(f"[OK] Saved per-sample external results: {RESULT_OUT}")
    print(f"[OK] Saved summary: {SUMMARY_OUT}")
    print("-" * 90)
    print(f"[INFO] Best params: {grid.best_params_}")
    print(f"[INFO] ROC_AUC: {metrics['ROC_AUC']:.4f}")
    print(f"[INFO] PR_AUC: {metrics['PR_AUC']:.4f}")
    print(f"[INFO] MCC: {metrics['MCC']:.4f}")
    print(f"[INFO] F1: {metrics['F1']:.4f}")
    print(f"[INFO] Balanced_Accuracy: {metrics['Balanced_Accuracy']:.4f}")
    print(f"[INFO] Confusion matrix: TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}, TP={metrics['TP']}")
    print("=" * 90)
    print("[DONE] External independent test finished.")
    print("=" * 90)


if __name__ == "__main__":
    main()