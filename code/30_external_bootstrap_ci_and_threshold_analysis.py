from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "25_external_test_results.csv"
CI_OUT = BASE_DIR / "30_external_bootstrap_ci.csv"
THRESHOLD_OUT = BASE_DIR / "30_external_threshold_sensitivity.csv"

RANDOM_STATE = 42
N_BOOTSTRAP = 10000
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def compute_metrics(y_true, y_prob, y_pred) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
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


def bootstrap_ci(y_true, y_prob, y_pred) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    values = {
        "ROC_AUC": [],
        "PR_AUC": [],
        "MCC": [],
        "F1": [],
        "Balanced_Accuracy": [],
        "Precision": [],
        "Recall": [],
    }

    valid = 0
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yy = y_true[idx]
        if len(np.unique(yy)) < 2:
            continue
        pp = y_prob[idx]
        yh = y_pred[idx]
        m = compute_metrics(yy, pp, yh)
        for metric in values:
            values[metric].append(m[metric])
        valid += 1

    rows: List[Dict] = []
    observed = compute_metrics(y_true, y_prob, y_pred)
    for metric, vals in values.items():
        arr = np.array(vals, dtype=float)
        rows.append(
            {
                "Metric": metric,
                "Observed": observed[metric],
                "Bootstrap_Mean": float(np.mean(arr)),
                "CI95_Lower": float(np.percentile(arr, 2.5)),
                "CI95_Upper": float(np.percentile(arr, 97.5)),
                "Valid_Resamples": valid,
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity(y_true, y_prob) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append({"Threshold": threshold, **compute_metrics(y_true, y_prob, y_pred)})
    return pd.DataFrame(rows)


def main():
    print("=" * 90)
    print("Step 30: external bootstrap confidence intervals and descriptive threshold sensitivity")
    print("=" * 90)
    print(f"[INFO] Input CSV: {INPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Required file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    y_true = df["y_true"].astype(int).to_numpy()
    y_prob = df["y_prob"].astype(float).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()

    print(f"[INFO] External N={len(y_true)}, Pos={int(np.sum(y_true))}, Neg={int(len(y_true) - np.sum(y_true))}")
    observed = compute_metrics(y_true, y_prob, y_pred)
    print("[INFO] Observed metrics:")
    for key, value in observed.items():
        print(f"  {key}: {value}")

    ci_df = bootstrap_ci(y_true, y_prob, y_pred)
    threshold_df = threshold_sensitivity(y_true, y_prob)

    ci_df.to_csv(CI_OUT, index=False)
    threshold_df.to_csv(THRESHOLD_OUT, index=False)

    print(f"[OK] Saved bootstrap CI: {CI_OUT}")
    print(ci_df.to_string(index=False))
    print(f"[OK] Saved threshold sensitivity: {THRESHOLD_OUT}")
    print(threshold_df.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
