from pathlib import Path
from typing import Dict, List
import gc
import warnings

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
STRICT_FEATURES = BASE_DIR / "15_strict_nomixed_esm2_features.csv"
REMOVED_CSV = BASE_DIR / "13_mixed_label_representatives_removed.csv"
REMOVED_FEATURES_OUT = BASE_DIR / "31_mixed_label_removed_representatives_esm2_150m_features.csv"
FOLD_OUT = BASE_DIR / "31_mixed_label_sensitivity_fold_results.csv"
SUMMARY_OUT = BASE_DIR / "31_mixed_label_sensitivity_summary.csv"

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
MAX_LENGTH = 800
BATCH_SIZE = 4
LOCAL_FILES_ONLY = True
USE_FP16 = True
RANDOM_STATE = 42

HASH_COL = "Sequence_Hash"
LABEL_COL = "Binary_Label"
SEQ_COL = "Protein_Sequence"
TEMP_COL = "Temp_Median"


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_strict_features():
    df = pd.read_csv(STRICT_FEATURES)
    feat_cols = [c for c in df.columns if c.startswith("ESM_")]
    keep = [HASH_COL, "Target_Temperature", LABEL_COL] + feat_cols
    df = df[keep].drop_duplicates(subset=[HASH_COL]).copy()
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").astype(int)
    return df, feat_cols


def extract_removed_features(feat_cols: List[str]) -> pd.DataFrame:
    if REMOVED_FEATURES_OUT.exists():
        existing = pd.read_csv(REMOVED_FEATURES_OUT)
        if all(c in existing.columns for c in feat_cols):
            print(f"[INFO] Using cached removed representative features: {REMOVED_FEATURES_OUT}")
            return existing

    removed = pd.read_csv(REMOVED_CSV)
    removed = removed.dropna(subset=[HASH_COL, SEQ_COL, LABEL_COL]).drop_duplicates(subset=[HASH_COL]).copy()
    removed[LABEL_COL] = pd.to_numeric(removed[LABEL_COL], errors="coerce").astype(int)
    print(f"[INFO] Removed representative rows to embed: {len(removed)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)
    model = EsmModel.from_pretrained(MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)
    model.to(device)
    model.eval()

    rows: List[Dict] = []
    batches = (len(removed) + BATCH_SIZE - 1) // BATCH_SIZE
    for start in tqdm(range(0, len(removed), BATCH_SIZE), total=batches, desc="Removed mixed-label embeddings"):
        batch = removed.iloc[start:start + BATCH_SIZE].copy()
        sequences = batch[SEQ_COL].astype(str).str.strip().str.upper().tolist()
        inputs = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available() and USE_FP16):
                outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
        for i, row in enumerate(batch.itertuples(index=False)):
            out = {
                HASH_COL: str(getattr(row, HASH_COL)),
                "Target_Temperature": float(getattr(row, TEMP_COL)),
                LABEL_COL: int(getattr(row, LABEL_COL)),
            }
            for j, val in enumerate(embeddings[i]):
                out[f"ESM_{j}"] = float(val)
            rows.append(out)
        del inputs, outputs
        clear_memory()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(REMOVED_FEATURES_OUT, index=False)
    print(f"[OK] Saved removed representative features: {REMOVED_FEATURES_OUT}")
    return out_df


def compute_metrics(y_true, y_prob, y_pred) -> Dict[str, float]:
    return {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def eval_fixed_rf(df: pd.DataFrame, feat_cols: List[str], dataset_name: str) -> List[Dict]:
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    records = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        pipe = ImbPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=20,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipe.fit(X[train_idx], y[train_idx])
        y_prob = pipe.predict_proba(X[test_idx])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_metrics(y[test_idx], y_prob, y_pred)
        record = {
            "Dataset": dataset_name,
            "Fold": fold,
            "N_Train": len(train_idx),
            "N_Test": len(test_idx),
            "Pos_Train": int(np.sum(y[train_idx])),
            "Pos_Test": int(np.sum(y[test_idx])),
            **metrics,
        }
        records.append(record)
        print(
            f"[{dataset_name}] Fold {fold} | "
            f"ROC-AUC={metrics['ROC_AUC']:.4f}, PR-AUC={metrics['PR_AUC']:.4f}, "
            f"MCC={metrics['MCC']:.4f}, F1={metrics['F1']:.4f}, "
            f"BA={metrics['Balanced_Accuracy']:.4f}"
        )
    return records


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in fold_df.groupby("Dataset", sort=False):
        row = {
            "Dataset": name,
            "N_Samples": int(group["N_Test"].sum()),
            "Positives": int(group["Pos_Test"].sum()),
            "Negatives": int(group["N_Test"].sum() - group["Pos_Test"].sum()),
        }
        for metric in ["ROC_AUC", "PR_AUC", "MCC", "F1", "Balanced_Accuracy"]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("=" * 100)
    print("Step 31: mixed-label cluster sensitivity analysis")
    print("=" * 100)

    strict_df, feat_cols = load_strict_features()
    removed_features = extract_removed_features(feat_cols)
    combined_df = pd.concat([strict_df, removed_features], ignore_index=True).drop_duplicates(subset=[HASH_COL]).copy()
    combined_df[LABEL_COL] = pd.to_numeric(combined_df[LABEL_COL], errors="coerce").astype(int)

    print(f"[INFO] No-mixed rows: {len(strict_df)}, positives: {int(strict_df[LABEL_COL].sum())}")
    print(f"[INFO] Mixed-retained rows: {len(combined_df)}, positives: {int(combined_df[LABEL_COL].sum())}")

    records = []
    records.extend(eval_fixed_rf(strict_df, feat_cols, "strict_no_mixed_N5263"))
    records.extend(eval_fixed_rf(combined_df, feat_cols, "mixed_retained_representative_label_N5568"))

    fold_df = pd.DataFrame(records)
    summary_df = summarize(fold_df)
    fold_df.to_csv(FOLD_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print(f"[OK] Saved fold results: {FOLD_OUT}")
    print(f"[OK] Saved summary: {SUMMARY_OUT}")
    print(summary_df.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
