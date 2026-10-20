from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "15_strict_nomixed_esm2_features.csv"
SUMMARY_OUT = BASE_DIR / "26_esm2_topt_regression_summary.csv"
FOLD_OUT = BASE_DIR / "26_esm2_topt_regression_fold_results.csv"

TEMP_COL = "Target_Temperature"
RANDOM_STATE = 42
N_SPLITS = 5


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if np.std(y_pred) == 0:
        pearson = np.nan
        spearman = np.nan
    else:
        pearson = float(pearsonr(y_true, y_pred)[0])
        spearman = float(spearmanr(y_true, y_pred).correlation)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": pearson,
        "Spearman": spearman,
    }


def build_models():
    return [
        ("DummyMean", DummyRegressor(strategy="mean")),
        (
            "RidgeCV",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                    ("reg", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
                ]
            ),
        ),
        (
            "RandomForest",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "ExtraTrees",
            ExtraTreesRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]


def summarize_fold_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict] = []
    for model_name, group in fold_df.groupby("Model", sort=False):
        record = {
            "Model": model_name,
            "N_Samples": int(group["N_Test"].sum()),
        }
        for metric in ["MAE", "RMSE", "R2", "Pearson", "Spearman"]:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_std"] = group[metric].std(ddof=1)
        records.append(record)
    return pd.DataFrame(records)


def main():
    print("=" * 90)
    print("Step 26: continuous Topt regression baseline from frozen ESM-2 150M embeddings")
    print("=" * 90)
    print(f"[INFO] Input CSV: {INPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Required file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    feature_cols = [c for c in df.columns if c.startswith("ESM_")]
    if not feature_cols:
        raise ValueError("No ESM feature columns found.")
    if TEMP_COL not in df.columns:
        raise ValueError(f"Missing target column: {TEMP_COL}")

    y = pd.to_numeric(df[TEMP_COL], errors="coerce").to_numpy(dtype=float)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    mask = np.isfinite(y)
    X = X[mask]
    y = y[mask]

    print(f"[INFO] Usable rows: {len(y)}")
    print(f"[INFO] ESM dimensions: {len(feature_cols)}")
    print(f"[INFO] Temperature min/median/max: {np.min(y):.2f}/{np.median(y):.2f}/{np.max(y):.2f}")

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_records: List[Dict] = []

    for model_name, model in build_models():
        print("-" * 90)
        print(f"[INFO] Evaluating: {model_name}")
        for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            metrics = compute_metrics(y[test_idx], pred)
            record = {
                "Model": model_name,
                "Fold": fold,
                "N_Train": len(train_idx),
                "N_Test": len(test_idx),
                **metrics,
            }
            fold_records.append(record)
            print(
                f"[{model_name}] Fold {fold} | "
                f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
                f"R2={metrics['R2']:.4f}, Pearson={metrics['Pearson']:.4f}, "
                f"Spearman={metrics['Spearman']:.4f}"
            )

    fold_df = pd.DataFrame(fold_records)
    summary_df = summarize_fold_results(fold_df)
    fold_df.to_csv(FOLD_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print("=" * 90)
    print(f"[OK] Saved fold results: {FOLD_OUT}")
    print(f"[OK] Saved summary: {SUMMARY_OUT}")
    print(summary_df.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
