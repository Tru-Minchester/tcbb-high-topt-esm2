from pathlib import Path
from typing import Dict, List
import gc
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "14_strict_nomixed_clean_dataset.csv"
OUTPUT_CSV = BASE_DIR / "27_strict_nomixed_esm2_650m_features.csv"

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_LENGTH = 800
BATCH_SIZE = 1
SAVE_EVERY_BATCHES = 20
USE_FP16 = True
LOCAL_FILES_ONLY = True

ID_COL = "UniProt_Accession"
HASH_COL = "Sequence_Hash"
TEMP_COL = "Target_Temperature"
LABEL_COL = "Binary_Label"
SEQ_COL = "Protein_Sequence"
LEN_COL = "Sequence_Length"


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


def build_output_rows(batch_df: pd.DataFrame, cls_embeddings, max_length: int) -> List[Dict]:
    rows = []
    for i, row in enumerate(batch_df.itertuples(index=False)):
        accession = str(getattr(row, ID_COL))
        seq_hash = str(getattr(row, HASH_COL))
        target_temp = float(getattr(row, TEMP_COL))
        binary_label = int(getattr(row, LABEL_COL))
        seq_len = int(getattr(row, LEN_COL))

        out_row = {
            "UniProt_Accession": accession,
            "Sequence_Hash": seq_hash,
            "Target_Temperature": target_temp,
            "Binary_Label": binary_label,
            "Sequence_Length": seq_len,
            "Was_Truncated": int(seq_len > (max_length - 2)),
        }
        for j, val in enumerate(cls_embeddings[i]):
            out_row[f"ESM650_{j}"] = float(val)
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


def main():
    print("=" * 90)
    print("Step 27: extract ESM-2 650M embeddings from strict no-mixed dataset")
    print("=" * 90)
    print(f"[INFO] Input CSV: {INPUT_CSV}")
    print(f"[INFO] Output CSV: {OUTPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Required file not found: {INPUT_CSV}")

    device = get_device()
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Torch CUDA: {torch.version.cuda}")
        print(f"[INFO] VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    df = pd.read_csv(INPUT_CSV)
    required_cols = [ID_COL, HASH_COL, TEMP_COL, LABEL_COL, SEQ_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required_cols).copy()
    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()
    if LEN_COL not in df.columns:
        df[LEN_COL] = df[SEQ_COL].str.len()
    else:
        df[LEN_COL] = pd.to_numeric(df[LEN_COL], errors="coerce").fillna(df[SEQ_COL].str.len()).astype(int)

    finished_hashes = load_finished_hashes(OUTPUT_CSV)
    if finished_hashes:
        df = df[~df[HASH_COL].astype(str).isin(finished_hashes)].copy()
        print(f"[INFO] Resume mode: {len(finished_hashes)} rows already saved.")
        print(f"[INFO] Remaining rows: {len(df)}")

    if len(df) == 0:
        print("[INFO] No rows left to process.")
        return

    print(f"[INFO] Total rows to process: {len(df)}")
    print(f"[INFO] MAX_LENGTH: {MAX_LENGTH}")
    print(f"[INFO] BATCH_SIZE: {BATCH_SIZE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)
    model = EsmModel.from_pretrained(MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)
    model.to(device)
    model.eval()

    use_autocast = torch.cuda.is_available() and USE_FP16
    rows_buffer: List[Dict] = []
    processed = 0
    skipped = 0
    start_time = time.time()

    batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_id, batch_df in enumerate(tqdm(batch_iter(df, BATCH_SIZE), total=batches, desc="ESM2-650M batches"), start=1):
        sequences = batch_df[SEQ_COL].astype(str).tolist()
        inputs = None
        outputs = None
        try:
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
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_autocast):
                    outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
            rows_buffer.extend(build_output_rows(batch_df, cls_embeddings, MAX_LENGTH))
            processed += len(batch_df)
        except torch.cuda.OutOfMemoryError:
            skipped += len(batch_df)
            print("\n[WARN] CUDA OOM, batch skipped. Keep BATCH_SIZE=1 and consider reducing MAX_LENGTH.")
            clear_memory()
        except Exception as exc:
            skipped += len(batch_df)
            print(f"\n[WARN] Batch skipped due to error: {exc}")
            clear_memory()
        finally:
            del inputs
            if outputs is not None:
                del outputs
            clear_memory()

        if batch_id % SAVE_EVERY_BATCHES == 0:
            save_rows(rows_buffer, OUTPUT_CSV)
            rows_buffer = []

    save_rows(rows_buffer, OUTPUT_CSV)
    elapsed = time.time() - start_time
    print("=" * 90)
    print(f"[DONE] Output saved: {OUTPUT_CSV}")
    print(f"[INFO] Processed rows: {processed}")
    print(f"[INFO] Skipped rows: {skipped}")
    print(f"[INFO] Time elapsed: {elapsed / 60:.2f} min")
    print("=" * 90)


if __name__ == "__main__":
    main()
