from pathlib import Path
from typing import Dict, Tuple
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


warnings.filterwarnings("ignore")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PACKAGE_ROOT / "code"
PROJECT_DIR = PACKAGE_ROOT

TRAIN_CSV = PROJECT_DIR / "features" / "15_strict_nomixed_esm2_features.csv"

EXTERNALS = [
    ("Seq2Topt_no_exact_overlap", PROJECT_DIR / "features" / "32_external_noexact_esm2_150m_features.csv"),
    (
        "biomap_optimal_temperature_no_exact_overlap",
        PROJECT_DIR / "features" / "34_public_topt_noexact_esm2_150m_features.csv",
    ),
]

FOLD_OUT = PACKAGE_ROOT / "results" / "36_classifier_sensitivity_internal_fold_results.csv"
SUMMARY_OUT = PACKAGE_ROOT / "results" / "36_classifier_sensitivity_internal_summary.csv"
EXTERNAL_OUT = PACKAGE_ROOT / "results" / "36_classifier_sensitivity_external_summary.csv"

HASH_COL = "Sequence_Hash"
LABEL_COL = "Binary_Label"
FEATURE_PREFIX = "ESM_"
RANDOM_STATE = 42
N_SPLITS = 5


def build_classifiers() -> Dict[str, object]:
    return {
        "RandomForest_fixed": RandomForestClassifier(
            n_estimators=400,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "LinearSVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(clf):
    return ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", clf),
    ])


def get_score(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def compute_metrics(y_true, y_score, y_pred) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, y_score),
        "PR_AUC": average_precision_score(y_true, y_score),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def load_feature_table(path: Path) -> Tuple[pd.DataFrame, list]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    if not feature_cols:
        raise ValueError(f"No {FEATURE_PREFIX} feature columns in {path}")
    feature_cols = sorted(feature_cols)
    df = df.dropna(subset=[HASH_COL, LABEL_COL]).drop_duplicates(subset=[HASH_COL]).copy()
    return df, feature_cols


def main():
    print("=" * 90)
    print("Step 36: classifier sensitivity using frozen ESM-2 150M embeddings")
    print("=" * 90)

    train_df, feature_cols = load_feature_table(TRAIN_CSV)
    X = train_df[feature_cols].values
    y = train_df[LABEL_COL].astype(int).values

    print(f"[INFO] Train samples: {len(train_df)}")
    print(f"[INFO] Feature dim: {len(feature_cols)}")
    print(f"[INFO] Positives: {int(np.sum(y))}")
    print(f"[INFO] Negatives: {int(len(y) - np.sum(y))}")

    classifiers = build_classifiers()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_rows = []
    for clf_name, clf in classifiers.items():
        print("-" * 90)
        print(f"[INFO] Internal CV classifier: {clf_name}")
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            pipe = build_pipeline(clf)
            pipe.fit(X[train_idx], y[train_idx])
            y_score = get_score(pipe, X[test_idx])
            y_pred = pipe.predict(X[test_idx])
            metrics = compute_metrics(y[test_idx], y_score, y_pred)
            fold_rows.append({
                "Classifier": clf_name,
                "Fold": fold,
                "N_Test": int(len(test_idx)),
                "Positives_Test": int(np.sum(y[test_idx])),
                **metrics,
            })

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(FOLD_OUT, index=False)

    metric_cols = ["ROC_AUC", "PR_AUC", "MCC", "F1", "Balanced_Accuracy"]
    summary_rows = []
    for clf_name, group in fold_df.groupby("Classifier", sort=False):
        row = {"Classifier": clf_name, "N_Samples": len(train_df), "Feature_Dim": len(feature_cols)}
        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    external_rows = []
    for clf_name, clf in classifiers.items():
        print("-" * 90)
        print(f"[INFO] External evaluation classifier: {clf_name}")
        pipe = build_pipeline(clf)
        pipe.fit(X, y)

        for external_name, external_csv in EXTERNALS:
            ext_df, ext_cols = load_feature_table(external_csv)
            common_cols = sorted(set(feature_cols) & set(ext_cols))
            X_ext = ext_df[common_cols].values
            y_ext = ext_df[LABEL_COL].astype(int).values
            y_score = get_score(pipe, X_ext)
            y_pred = pipe.predict(X_ext)
            metrics = compute_metrics(y_ext, y_score, y_pred)
            external_rows.append({
                "Classifier": clf_name,
                "External_Set": external_name,
                "External_N": len(ext_df),
                "External_Pos": int(np.sum(y_ext)),
                "External_Neg": int(len(y_ext) - np.sum(y_ext)),
                **metrics,
            })

    external_df = pd.DataFrame(external_rows)
    external_df.to_csv(EXTERNAL_OUT, index=False)

    print("-" * 90)
    print(f"[OK] Saved fold results: {FOLD_OUT}")
    print(f"[OK] Saved internal summary: {SUMMARY_OUT}")
    print(f"[OK] Saved external summary: {EXTERNAL_OUT}")
    print(summary_df.to_string(index=False))
    print(external_df.to_string(index=False))
    print("=" * 90)
    print("[DONE] Classifier sensitivity finished.")
    print("=" * 90)


if __name__ == "__main__":
    main()
