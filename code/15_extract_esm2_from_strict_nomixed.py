from pathlib import Path
from typing import List, Dict
import gc
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "14_strict_nomixed_clean_dataset.csv"
OUTPUT_CSV = BASE_DIR / "15_strict_nomixed_esm2_features.csv"

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"

# 你的 5060 + CUDA 环境建议先用这个配置
MAX_LENGTH = 800
BATCH_SIZE = 4
SAVE_EVERY_BATCHES = 10
USE_FP16 = True

ID_COL = "UniProt_Accession"
HASH_COL = "Sequence_Hash"
TEMP_COL = "Target_Temperature"
LABEL_COL = "Binary_Label"
SEQ_COL = "Protein_Sequence"
LEN_COL = "Sequence_Length"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    n = len(df)
    for start in range(0, n, batch_size):
        yield df.iloc[start:start + batch_size].copy()


def build_output_rows(
    batch_df: pd.DataFrame,
    cls_embeddings,
    max_length: int,
) -> List[Dict]:
    rows = []
    for i, row in enumerate(batch_df.itertuples(index=False)):
        accession = str(getattr(row, ID_COL))
        seq_hash = str(getattr(row, HASH_COL))
        target_temp = float(getattr(row, TEMP_COL))
        binary_label = int(getattr(row, LABEL_COL))
        sequence = str(getattr(row, SEQ_COL))
        seq_len = int(getattr(row, LEN_COL))

        out_row = {
            "UniProt_Accession": accession,
            "Sequence_Hash": seq_hash,
            "Target_Temperature": target_temp,
            "Binary_Label": binary_label,
            "Sequence_Length": seq_len,
            "Was_Truncated": int(seq_len > (max_length - 2)),
        }

        emb = cls_embeddings[i]
        for j, val in enumerate(emb):
            out_row[f"ESM_{j}"] = float(val)

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
    print("=" * 80)
    print("Step 15: Extract ESM-2 embeddings from strict no-mixed dataset (CUDA)")
    print("=" * 80)
    print(f"[INFO] Input CSV: {INPUT_CSV}")
    print(f"[INFO] Output CSV: {OUTPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"找不到文件: {INPUT_CSV}")

    device = get_device()
    print(f"[INFO] Device: {device}")

    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA version (torch): {torch.version.cuda}")
        print(f"[INFO] VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    df = pd.read_csv(INPUT_CSV)

    required_cols = [ID_COL, HASH_COL, TEMP_COL, LABEL_COL, SEQ_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")

    df = df.dropna(subset=required_cols).copy()
    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()

    if LEN_COL not in df.columns:
        df[LEN_COL] = df[SEQ_COL].str.len()
    else:
        df[LEN_COL] = pd.to_numeric(df[LEN_COL], errors="coerce").fillna(df[SEQ_COL].str.len()).astype(int)

    finished_hashes = load_finished_hashes(OUTPUT_CSV)
    if finished_hashes:
        df = df[~df[HASH_COL].astype(str).isin(finished_hashes)].copy()
        print(f"[INFO] Resume mode detected.")
        print(f"[INFO] Remaining rows after resume filter: {len(df)}")

    if len(df) == 0:
        print("[INFO] 没有剩余样本需要处理。")
        return

    print(f"[INFO] Total rows to process: {len(df)}")
    print(f"[INFO] MAX_LENGTH: {MAX_LENGTH}")
    print(f"[INFO] BATCH_SIZE: {BATCH_SIZE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    use_autocast = torch.cuda.is_available() and USE_FP16

    rows_buffer: List[Dict] = []
    processed = 0
    skipped = 0
    batch_counter = 0
    start_time = time.time()

    for batch_df in tqdm(
        batch_iter(df, BATCH_SIZE),
        total=(len(df) + BATCH_SIZE - 1) // BATCH_SIZE,
        desc="Embedding batches"
    ):
        sequences = batch_df[SEQ_COL].astype(str).tolist()

        try:
            inputs = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                add_special_tokens=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                if use_autocast:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
            batch_rows = build_output_rows(batch_df, cls_embeddings, MAX_LENGTH)
            rows_buffer.extend(batch_rows)
            processed += len(batch_rows)

        except torch.cuda.OutOfMemoryError:
            skipped += len(batch_df)
            print("\n[WARN] CUDA 显存不足，本批次已跳过。")
            print(f"[WARN] 建议把 BATCH_SIZE 从 {BATCH_SIZE} 改成 {max(1, BATCH_SIZE // 2)} 后重跑。")
            clear_memory()
            continue

        except Exception as e:
            skipped += len(batch_df)
            print(f"\n[WARN] 本批次跳过，错误: {e}")
            clear_memory()
            continue

        finally:
            del inputs
            if 'outputs' in locals():
                del outputs
            clear_memory()

        batch_counter += 1
        if batch_counter % SAVE_EVERY_BATCHES == 0:
            save_rows(rows_buffer, OUTPUT_CSV)
            rows_buffer = []

    if rows_buffer:
        save_rows(rows_buffer, OUTPUT_CSV)

    elapsed = time.time() - start_time

    print("=" * 80)
    print(f"[DONE] Output saved: {OUTPUT_CSV}")
    print(f"[INFO] Processed rows: {processed}")
    print(f"[INFO] Skipped rows: {skipped}")
    print(f"[INFO] Time elapsed: {elapsed / 60:.2f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()