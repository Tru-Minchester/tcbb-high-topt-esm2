from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


BASE_DIR = Path(__file__).resolve().parent

INPUTS = [
    ("Seq2Topt_no_exact_overlap", BASE_DIR / "32_external_noexact_model_scale_results.csv"),
    ("biomap_optimal_temperature_no_exact_overlap", BASE_DIR / "34_public_topt_noexact_model_scale_results.csv"),
]

OUT_CSV = BASE_DIR / "35_expanded_external_threshold_sensitivity.csv"
THRESHOLDS = [round(x, 2) for x in np.arange(0.1, 0.91, 0.1)]


def metrics_at_threshold(y_true, y_prob, threshold: float) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Threshold": threshold,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main():
    rows = []
    for dataset_name, path in INPUTS:
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        for model_name, group in df.groupby("Embedding_Model", sort=False):
            y_true = group["y_true"].astype(int).values
            y_prob = group["y_prob"].astype(float).values
            for threshold in THRESHOLDS:
                row = {
                    "External_Set": dataset_name,
                    "Embedding_Model": model_name,
                    "N": len(group),
                    "Positives": int(np.sum(y_true)),
                    "Negatives": int(len(y_true) - np.sum(y_true)),
                    **metrics_at_threshold(y_true, y_prob, threshold),
                }
                rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved threshold sensitivity: {OUT_CSV}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
