from pathlib import Path
from typing import Dict, List
import gc
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

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
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


warnings.filterwarnings("ignore")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PACKAGE_ROOT / "code"
PROJECT_DIR = PACKAGE_ROOT

NOEXACT_CSV = PROJECT_DIR / "data_processed" / "20_external_clean_no_exact_overlap.csv"

MODEL_RUNS = [
    {
        "name": "ESM2_150M",
        "model_name": "facebook/esm2_t30_150M_UR50D",
        "train_csv": PROJECT_DIR / "features" / "15_strict_nomixed_esm2_features.csv",
        "external_features_csv": PACKAGE_ROOT / "features" / "32_external_noexact_esm2_150m_features.csv",
        "feature_prefix": "ESM_",
        "batch_size": 8,
        "rf_params": {
            "n_estimators": 400,
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
        },
    },
    {
        "name": "ESM2_650M",
        "model_name": "facebook/esm2_t33_650M_UR50D",
        "train_csv": PROJECT_DIR / "features" / "27_strict_nomixed_esm2_650m_features.csv",
        "external_features_csv": PACKAGE_ROOT / "features" / "32_external_noexact_esm2_650m_features.csv",
        "feature_prefix": "ESM650_",
        "batch_size": 1,
        "rf_params": {
            "n_estimators": 400,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
        },
    },
]

RESULT_OUT = PACKAGE_ROOT / "results" / "32_external_noexact_model_scale_results.csv"
SUMMARY_OUT = PACKAGE_ROOT / "results" / "32_external_noexact_model_scale_summary.csv"
DATASET_SUMMARY_OUT = PACKAGE_ROOT / "results" / "32_external_noexact_dataset_summary.txt"

HASH_COL = "Sequence_Hash"
SEQ_COL = "Protein_Sequence"
TEMP_COL = "Temp_Median"
LABEL_COL = "Binary_Label"
ACC_COL = "Representative_Accession"
LEN_COL = "Sequence_Length"

MAX_LENGTH = 800
SAVE_EVERY_BATCHES = 5
USE_FP16 = True
RANDOM_STATE = 42


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_finished_hashes(output_csv: Path) -> set:
    if not output_csv.exists():
        return set()
    try:
        old_df = pd.read_csv(output_csv, usecols=[HASH_COL])
        return set(old_df[HASH_COL].astype(str).tolist())
    except Exception:
        return set()


def batch_iter(df: pd.DataFrame, batch_size: int):
    for start in range(0, len(df), batch_size):
        yield df.iloc[start:start + batch_size].copy()


def prepare_noexact_df() -> pd.DataFrame:
    if not NOEXACT_CSV.exists():
        raise FileNotFoundError(f"File not found: {NOEXACT_CSV}")

    df = pd.read_csv(NOEXACT_CSV)
    required_cols = [HASH_COL, SEQ_COL, TEMP_COL, LABEL_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in no-exact external set: {missing}")

    if ACC_COL not in df.columns:
        df[ACC_COL] = "NA"
    if LEN_COL not in df.columns:
        df[LEN_COL] = df[SEQ_COL].astype(str).str.len()

    df = df.dropna(subset=[HASH_COL, SEQ_COL, TEMP_COL, LABEL_COL]).copy()
    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=[HASH_COL]).reset_index(drop=True)
    return df


def build_embedding_rows(batch_df: pd.DataFrame, cls_embeddings, feature_prefix: str) -> List[Dict]:
    rows = []
    for i, row in enumerate(batch_df.itertuples(index=False)):
        seq_len = int(getattr(row, LEN_COL))
        out_row = {
            "UniProt_Accession": str(getattr(row, ACC_COL, "NA")),
            "Sequence_Hash": str(getattr(row, HASH_COL)),
            "Target_Temperature": float(getattr(row, TEMP_COL)),
            "Binary_Label": int(getattr(row, LABEL_COL)),
            "Sequence_Length": seq_len,
            "Was_Truncated": int(seq_len > (MAX_LENGTH - 2)),
        }
        for j, val in enumerate(cls_embeddings[i]):
            out_row[f"{feature_prefix}{j}"] = float(val)
        rows.append(out_row)
    return rows


def save_rows(rows_buffer: List[Dict], output_csv: Path):
    if not rows_buffer:
        return
    out_df = pd.DataFrame(rows_buffer)
    if output_csv.exists():
        out_df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        out_df.to_csv(output_csv, index=False)


def extract_embeddings(run_cfg: Dict, source_df: pd.DataFrame, device: torch.device):
    output_csv = run_cfg["external_features_csv"]
    finished_hashes = load_finished_hashes(output_csv)
    df = source_df.copy()
    if finished_hashes:
        df = df[~df[HASH_COL].astype(str).isin(finished_hashes)].copy()

    print("-" * 90)
    print(f"[INFO] Extracting {run_cfg['name']} features")
    print(f"[INFO] Model: {run_cfg['model_name']}")
    print(f"[INFO] Output CSV: {output_csv}")
    print(f"[INFO] Rows to process: {len(df)}")
    print(f"[INFO] Existing rows skipped by resume: {len(source_df) - len(df)}")

    if len(df) == 0:
        return

    tokenizer = AutoTokenizer.from_pretrained(run_cfg["model_name"], local_files_only=True)
    model = EsmModel.from_pretrained(run_cfg["model_name"], local_files_only=True)
    model.to(device)
    model.eval()

    use_autocast = torch.cuda.is_available() and USE_FP16
    rows_buffer: List[Dict] = []
    processed = 0
    skipped = 0
    batch_counter = 0
    start_time = time.time()

    for batch_df in tqdm(
        batch_iter(df, int(run_cfg["batch_size"])),
        total=(len(df) + int(run_cfg["batch_size"]) - 1) // int(run_cfg["batch_size"]),
        desc=f"{run_cfg['name']} no-exact batches",
    ):
        inputs = None
        outputs = None
        try:
            inputs = tokenizer(
                batch_df[SEQ_COL].astype(str).tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                add_special_tokens=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                if use_autocast:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
            rows_buffer.extend(build_embedding_rows(batch_df, cls_embeddings, run_cfg["feature_prefix"]))
            processed += len(batch_df)

        except torch.cuda.OutOfMemoryError:
            skipped += len(batch_df)
            print("\n[WARN] CUDA OOM, batch skipped.")
            clear_memory()
            continue
        except Exception as exc:
            skipped += len(batch_df)
            print(f"\n[WARN] Batch skipped due to error: {exc}")
            clear_memory()
            continue
        finally:
            del inputs
            if outputs is not None:
                del outputs
            clear_memory()

        batch_counter += 1
        if batch_counter % SAVE_EVERY_BATCHES == 0:
            save_rows(rows_buffer, output_csv)
            rows_buffer = []

    if rows_buffer:
        save_rows(rows_buffer, output_csv)

    elapsed = time.time() - start_time
    print(f"[DONE] {run_cfg['name']} processed={processed}, skipped={skipped}, time={elapsed / 60:.2f} min")

    del model
    del tokenizer
    clear_memory()


def build_pipeline(rf_params: Dict):
    pipe = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
            **rf_params,
        )),
    ])
    return pipe


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


def evaluate_one_model(run_cfg: Dict) -> Dict:
    train_df = pd.read_csv(run_cfg["train_csv"])
    ext_df = pd.read_csv(run_cfg["external_features_csv"])

    feature_cols = [c for c in train_df.columns if c.startswith(run_cfg["feature_prefix"])]
    ext_feature_cols = [c for c in ext_df.columns if c.startswith(run_cfg["feature_prefix"])]
    common_cols = sorted(set(feature_cols) & set(ext_feature_cols))
    if not common_cols:
        raise ValueError(f"No shared feature columns for {run_cfg['name']}")

    train_df = train_df.dropna(subset=[HASH_COL, LABEL_COL]).drop_duplicates(subset=[HASH_COL]).copy()
    ext_df = ext_df.dropna(subset=[HASH_COL, LABEL_COL]).drop_duplicates(subset=[HASH_COL]).copy()

    X_train = train_df[common_cols].values
    y_train = train_df[LABEL_COL].astype(int).values
    X_ext = ext_df[common_cols].values
    y_ext = ext_df[LABEL_COL].astype(int).values

    pipe = build_pipeline(run_cfg["rf_params"])
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_ext)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_ext, y_prob, y_pred)

    result_rows = []
    for seq_hash, y_t, prob, pred in zip(ext_df[HASH_COL].astype(str), y_ext, y_prob, y_pred):
        result_rows.append({
            "Embedding_Model": run_cfg["name"],
            "Sequence_Hash": seq_hash,
            "y_true": int(y_t),
            "y_prob": float(prob),
            "y_pred": int(pred),
        })

    result_df = pd.DataFrame(result_rows)
    if RESULT_OUT.exists():
        old_df = pd.read_csv(RESULT_OUT)
        old_df = old_df[old_df["Embedding_Model"] != run_cfg["name"]].copy()
        result_df = pd.concat([old_df, result_df], ignore_index=True)
    result_df.to_csv(RESULT_OUT, index=False)

    return {
        "Embedding_Model": run_cfg["name"],
        "Feature_Dim": len(common_cols),
        "Train_N": len(train_df),
        "Train_Pos": int(np.sum(y_train)),
        "Train_Neg": int(len(y_train) - np.sum(y_train)),
        "External_Set": "Seq2Topt_no_exact_overlap",
        "External_N": len(ext_df),
        "External_Pos": int(np.sum(y_ext)),
        "External_Neg": int(len(y_ext) - np.sum(y_ext)),
        "Best_Params": json.dumps(run_cfg["rf_params"], ensure_ascii=False),
        **metrics,
    }


def main():
    print("=" * 90)
    print("Step 32: larger no-exact-overlap external validation")
    print("=" * 90)

    noexact_df = prepare_noexact_df()
    dataset_summary = [
        "Step 32 dataset summary",
        f"External source: {NOEXACT_CSV.name}",
        f"No-exact-overlap rows: {len(noexact_df)}",
        f"Positives: {int((noexact_df[LABEL_COL] == 1).sum())}",
        f"Negatives: {int((noexact_df[LABEL_COL] == 0).sum())}",
        f"Temperature range: {float(noexact_df[TEMP_COL].min()):.2f}-{float(noexact_df[TEMP_COL].max()):.2f} C",
        "Note: this set removes exact sequence overlap with the strict training benchmark but is not filtered by CD-HIT-2D at 40% identity.",
    ]
    DATASET_SUMMARY_OUT.write_text("\n".join(dataset_summary), encoding="utf-8")
    print("\n".join(dataset_summary))

    device = get_device()
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Torch CUDA: {torch.version.cuda}")

    for run_cfg in MODEL_RUNS:
        extract_embeddings(run_cfg, noexact_df, device)

    summary_rows = []
    for run_cfg in MODEL_RUNS:
        print("-" * 90)
        print(f"[INFO] Evaluating RF+SMOTE with {run_cfg['name']} features")
        summary_rows.append(evaluate_one_model(run_cfg))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print("-" * 90)
    print(f"[OK] Saved per-sample results: {RESULT_OUT}")
    print(f"[OK] Saved summary: {SUMMARY_OUT}")
    print(summary_df.to_string(index=False))
    print("=" * 90)
    print("[DONE] No-exact-overlap external validation finished.")
    print("=" * 90)


if __name__ == "__main__":
    main()
