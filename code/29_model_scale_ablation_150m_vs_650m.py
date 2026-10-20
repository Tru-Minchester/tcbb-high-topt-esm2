from pathlib import Path
from typing import Dict, List, Tuple
import json
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

FEATURE_FILES = {
    "ESM2_150M": BASE_DIR / "15_strict_nomixed_esm2_features.csv",
    "ESM2_650M": BASE_DIR / "27_strict_nomixed_esm2_650m_features.csv",
}
EXTERNAL_FILES = {
    "ESM2_150M": BASE_DIR / "24_external_esm2_features.csv",
    "ESM2_650M": BASE_DIR / "28_external_esm2_650m_features.csv",
}

INTERNAL_FOLD_OUT = BASE_DIR / "29_model_scale_internal_fold_results.csv"
INTERNAL_SUMMARY_OUT = BASE_DIR / "29_model_scale_internal_summary.csv"
EXTERNAL_RESULT_OUT = BASE_DIR / "29_model_scale_external_results.csv"
EXTERNAL_SUMMARY_OUT = BASE_DIR / "29_model_scale_external_summary.csv"

HASH_COL = "Sequence_Hash"
LABEL_COL = "Binary_Label"
RANDOM_STATE = 42
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 3


def feature_prefix(model_name: str) -> str:
    return "ESM650_" if model_name == "ESM2_650M" else "ESM_"


def load_feature_table(path: Path, model_name: str) -> Tuple[pd.DataFrame, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required feature file not found: {path}")
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith(feature_prefix(model_name))]
    if not feat_cols:
        raise ValueError(f"No feature columns found for {model_name} in {path}")
    required = [HASH_COL, LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    df = df.dropna(subset=required).drop_duplicates(subset=[HASH_COL]).copy()
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").astype(int)
    return df, feat_cols


def build_pipeline():
    pipe = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
        "clf__class_weight": [None, "balanced_subsample"],
    }
    return pipe, param_grid


def compute_classification_metrics(y_true, y_prob, y_pred) -> Dict[str, float]:
    return {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def evaluate_internal(model_name: str, df: pd.DataFrame, feat_cols: List[str]):
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=int)
    outer_cv = StratifiedKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=RANDOM_STATE)

    records: List[Dict] = []
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        pipe, param_grid = build_pipeline()
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid.fit(X[train_idx], y[train_idx])
        y_prob = grid.best_estimator_.predict_proba(X[test_idx])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_classification_metrics(y[test_idx], y_prob, y_pred)
        record = {
            "Embedding_Model": model_name,
            "Fold": fold,
            "Feature_Dim": len(feat_cols),
            "N_Train": len(train_idx),
            "N_Test": len(test_idx),
            "Pos_Train": int(np.sum(y[train_idx])),
            "Pos_Test": int(np.sum(y[test_idx])),
            "Best_Params": json.dumps(grid.best_params_, ensure_ascii=False),
            **metrics,
        }
        records.append(record)
        print(
            f"[{model_name}] Fold {fold} | "
            f"ROC-AUC={metrics['ROC_AUC']:.4f}, PR-AUC={metrics['PR_AUC']:.4f}, "
            f"MCC={metrics['MCC']:.4f}, F1={metrics['F1']:.4f}, "
            f"BA={metrics['Balanced_Accuracy']:.4f}"
        )
    return records


def summarize_internal(fold_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model_name, group in fold_df.groupby("Embedding_Model", sort=False):
        record = {
            "Embedding_Model": model_name,
            "Feature_Dim": int(group["Feature_Dim"].iloc[0]),
            "N_Samples": int(group["N_Test"].sum()),
            "Positives": int(group["Pos_Test"].sum()),
            "Negatives": int(group["N_Test"].sum() - group["Pos_Test"].sum()),
        }
        for metric in ["ROC_AUC", "PR_AUC", "MCC", "F1", "Balanced_Accuracy"]:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_std"] = group[metric].std(ddof=1)
        records.append(record)
    return pd.DataFrame(records)


def evaluate_external(model_name: str, train_df: pd.DataFrame, train_cols: List[str]):
    ext_path = EXTERNAL_FILES[model_name]
    ext_df, ext_cols = load_feature_table(ext_path, model_name)
    common_cols = sorted(set(train_cols) & set(ext_cols))
    if not common_cols:
        raise ValueError(f"No shared feature columns for {model_name}")

    train_df = train_df.drop_duplicates(subset=[HASH_COL]).copy()
    ext_df = ext_df.drop_duplicates(subset=[HASH_COL]).copy()

    X_train = train_df[common_cols].to_numpy(dtype=np.float32)
    y_train = train_df[LABEL_COL].to_numpy(dtype=int)
    X_ext = ext_df[common_cols].to_numpy(dtype=np.float32)
    y_ext = ext_df[LABEL_COL].to_numpy(dtype=int)

    pipe, param_grid = build_pipeline()
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    y_prob = grid.best_estimator_.predict_proba(X_ext)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_ext, y_pred, labels=[0, 1]).ravel()
    metrics = compute_classification_metrics(y_ext, y_prob, y_pred)

    per_sample = ext_df[[HASH_COL]].copy()
    per_sample["Embedding_Model"] = model_name
    per_sample["y_true"] = y_ext
    per_sample["y_prob"] = y_prob
    per_sample["y_pred"] = y_pred

    summary = {
        "Embedding_Model": model_name,
        "Feature_Dim": len(common_cols),
        "Train_N": len(train_df),
        "Train_Pos": int(np.sum(y_train)),
        "External_N": len(ext_df),
        "External_Pos": int(np.sum(y_ext)),
        "External_Neg": int(len(y_ext) - np.sum(y_ext)),
        "Best_Params": json.dumps(grid.best_params_, ensure_ascii=False),
        **metrics,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    print(
        f"[External {model_name}] ROC-AUC={metrics['ROC_AUC']:.4f}, "
        f"PR-AUC={metrics['PR_AUC']:.4f}, MCC={metrics['MCC']:.4f}, "
        f"F1={metrics['F1']:.4f}, BA={metrics['Balanced_Accuracy']:.4f}, "
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )
    return per_sample, summary


def main():
    print("=" * 100)
    print("Step 29: ESM-2 model-scale ablation, 150M vs 650M")
    print("=" * 100)

    all_internal_records: List[Dict] = []
    external_samples = []
    external_summaries = []
    loaded = {}

    for model_name, path in FEATURE_FILES.items():
        print("-" * 100)
        print(f"[INFO] Loading {model_name}: {path}")
        df, feat_cols = load_feature_table(path, model_name)
        loaded[model_name] = (df, feat_cols)
        pos = int(df[LABEL_COL].sum())
        print(f"[INFO] Rows={len(df)}, Positives={pos}, Negatives={len(df)-pos}, Dim={len(feat_cols)}")
        all_internal_records.extend(evaluate_internal(model_name, df, feat_cols))

    fold_df = pd.DataFrame(all_internal_records)
    summary_df = summarize_internal(fold_df)
    fold_df.to_csv(INTERNAL_FOLD_OUT, index=False)
    summary_df.to_csv(INTERNAL_SUMMARY_OUT, index=False)

    print("=" * 100)
    print("[INFO] Internal model-scale summary")
    print(summary_df.to_string(index=False))

    print("=" * 100)
    print("[INFO] External model-scale evaluation")
    for model_name, (df, feat_cols) in loaded.items():
        per_sample, summary = evaluate_external(model_name, df, feat_cols)
        external_samples.append(per_sample)
        external_summaries.append(summary)

    pd.concat(external_samples, ignore_index=True).to_csv(EXTERNAL_RESULT_OUT, index=False)
    pd.DataFrame(external_summaries).to_csv(EXTERNAL_SUMMARY_OUT, index=False)

    print(f"[OK] Saved internal fold results: {INTERNAL_FOLD_OUT}")
    print(f"[OK] Saved internal summary: {INTERNAL_SUMMARY_OUT}")
    print(f"[OK] Saved external per-sample results: {EXTERNAL_RESULT_OUT}")
    print(f"[OK] Saved external summary: {EXTERNAL_SUMMARY_OUT}")
    print("=" * 100)


if __name__ == "__main__":
    main()
