import os
import warnings
from typing import Dict, List, Tuple

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
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# =========================================================
# 固定输入文件
# =========================================================
CLEAN_CSV = "06_cdhit40_clean_dataset.csv"
AAC_CSV = "06_cdhit40_aac_features.csv"
DPC_CSV = "06_cdhit40_dpc_features.csv"
ESM_CSV = "07_cdhit40_esm2_features.csv"

TEMP_THRESHOLD = 60.0
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 3
RANDOM_STATE = 42

SUMMARY_OUT = "08_cdhit40_feature_set_summary.csv"
FOLD_OUT = "08_cdhit40_feature_set_fold_results.csv"


def check_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")


def binary_label(temp: float, threshold: float = 60.0) -> int:
    return int(float(temp) >= threshold)


def compute_metrics(y_true, y_prob, y_pred) -> Dict[str, float]:
    return {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def load_data():
    for p in [CLEAN_CSV, AAC_CSV, DPC_CSV, ESM_CSV]:
        check_file_exists(p)

    clean_df = pd.read_csv(CLEAN_CSV)
    aac_df = pd.read_csv(AAC_CSV)
    dpc_df = pd.read_csv(DPC_CSV)
    esm_df = pd.read_csv(ESM_CSV)

    required = {"Sequence_Hash", "Target_Temperature"}
    for name, df in [("clean", clean_df), ("aac", aac_df), ("dpc", dpc_df), ("esm", esm_df)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} 缺少列: {missing}")

    return clean_df, aac_df, dpc_df, esm_df


def merge_by_hash(clean_df, feat_df, prefixes: List[str]) -> pd.DataFrame:
    feat_cols = [c for c in feat_df.columns if any(c.startswith(p) for p in prefixes)]
    if len(feat_cols) == 0:
        raise ValueError(f"没有找到特征列，prefixes={prefixes}")

    merged = clean_df[["Sequence_Hash", "Target_Temperature"]].merge(
        feat_df[["Sequence_Hash"] + feat_cols],
        on="Sequence_Hash",
        how="inner"
    ).drop_duplicates(subset=["Sequence_Hash"])

    return merged


def build_pipeline():
    pipe = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])

    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
        "clf__class_weight": [None, "balanced_subsample"],
    }

    return pipe, param_grid


def evaluate_feature_set(df: pd.DataFrame, feature_cols: List[str], feature_name: str):
    X = df[feature_cols].values
    y = df["Target_Temperature"].apply(lambda x: binary_label(x, TEMP_THRESHOLD)).values

    outer_cv = StratifiedKFold(
        n_splits=N_SPLITS_OUTER,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    inner_cv = StratifiedKFold(
        n_splits=N_SPLITS_INNER,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_records = []

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe, param_grid = build_pipeline()

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            verbose=0
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_test, y_prob, y_pred)

        record = {
            "Feature_Set": feature_name,
            "Fold": fold_id,
            "Best_Params": str(grid.best_params_),
            "N_Train": len(train_idx),
            "N_Test": len(test_idx),
            "Pos_Train": int(np.sum(y_train)),
            "Pos_Test": int(np.sum(y_test)),
            **metrics
        }
        fold_records.append(record)

        print(
            f"[{feature_name}] Fold {fold_id} | "
            f"ROC-AUC={metrics['ROC_AUC']:.4f}, "
            f"PR-AUC={metrics['PR_AUC']:.4f}, "
            f"MCC={metrics['MCC']:.4f}, "
            f"F1={metrics['F1']:.4f}, "
            f"BA={metrics['Balanced_Accuracy']:.4f}"
        )

    fold_df = pd.DataFrame(fold_records)
    summary = {
        "Feature_Set": feature_name,
        "N_Samples": len(df),
        "Positives": int(np.sum(y)),
        "Negatives": int(len(y) - np.sum(y)),
        "ROC_AUC_mean": fold_df["ROC_AUC"].mean(),
        "ROC_AUC_std": fold_df["ROC_AUC"].std(ddof=1),
        "PR_AUC_mean": fold_df["PR_AUC"].mean(),
        "PR_AUC_std": fold_df["PR_AUC"].std(ddof=1),
        "MCC_mean": fold_df["MCC"].mean(),
        "MCC_std": fold_df["MCC"].std(ddof=1),
        "F1_mean": fold_df["F1"].mean(),
        "F1_std": fold_df["F1"].std(ddof=1),
        "Balanced_Accuracy_mean": fold_df["Balanced_Accuracy"].mean(),
        "Balanced_Accuracy_std": fold_df["Balanced_Accuracy"].std(ddof=1),
    }

    return fold_records, summary


def main():
    print("=" * 90)
    print("CD-HIT 40% 数据集：AAC / DPC / ESM-2 特征比较")
    print("=" * 90)

    clean_df, aac_df, dpc_df, esm_df = load_data()
    clean_df = clean_df.drop_duplicates(subset=["Sequence_Hash"]).copy()

    aac_merged = merge_by_hash(clean_df, aac_df, ["AAC_"])
    dpc_merged = merge_by_hash(clean_df, dpc_df, ["DPC_"])
    esm_merged = merge_by_hash(clean_df, esm_df, ["ESM_"])

    common_hashes = set(aac_merged["Sequence_Hash"]) & set(dpc_merged["Sequence_Hash"]) & set(esm_merged["Sequence_Hash"])
    print(f"[INFO] common rows = {len(common_hashes)}")

    hash_order = sorted(common_hashes)
    aac_merged = aac_merged[aac_merged["Sequence_Hash"].isin(common_hashes)].set_index("Sequence_Hash").loc[hash_order].reset_index()
    dpc_merged = dpc_merged[dpc_merged["Sequence_Hash"].isin(common_hashes)].set_index("Sequence_Hash").loc[hash_order].reset_index()
    esm_merged = esm_merged[esm_merged["Sequence_Hash"].isin(common_hashes)].set_index("Sequence_Hash").loc[hash_order].reset_index()

    aac_cols = [c for c in aac_merged.columns if c.startswith("AAC_")]
    dpc_cols = [c for c in dpc_merged.columns if c.startswith("DPC_")]
    esm_cols = [c for c in esm_merged.columns if c.startswith("ESM_")]

    print(f"[INFO] AAC dim = {len(aac_cols)}")
    print(f"[INFO] DPC dim = {len(dpc_cols)}")
    print(f"[INFO] ESM dim = {len(esm_cols)}")

    all_fold_records = []
    all_summaries = []

    for feature_name, df_feat, feat_cols in [
        ("AAC", aac_merged, aac_cols),
        ("DPC", dpc_merged, dpc_cols),
        ("ESM2", esm_merged, esm_cols),
    ]:
        print("-" * 90)
        print(f"[INFO] 正在评估: {feature_name}")
        fold_records, summary = evaluate_feature_set(df_feat, feat_cols, feature_name)
        all_fold_records.extend(fold_records)
        all_summaries.append(summary)

    fold_df = pd.DataFrame(all_fold_records)
    summary_df = pd.DataFrame(all_summaries)

    fold_df.to_csv(FOLD_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print("=" * 90)
    print(f"[OK] 保存: {FOLD_OUT}")
    print(f"[OK] 保存: {SUMMARY_OUT}")
    print("=" * 90)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()