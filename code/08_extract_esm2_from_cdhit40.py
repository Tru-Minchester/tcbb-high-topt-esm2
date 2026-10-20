import gc
import os
import time
from typing import List, Dict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

INPUT_CSV = "06_cdhit40_clean_dataset.csv"
OUTPUT_CSV = "07_cdhit40_esm2_features.csv"

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
MAX_LENGTH = 800
SAVE_EVERY = 100
USE_FP16 = True

ID_COL = "UniProt_Accession"
HASH_COL = "Sequence_Hash"
TEMP_COL = "Target_Temperature"
SEQ_COL = "Protein_Sequence"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_finished_hashes(output_csv: str) -> set:
    if not os.path.exists(output_csv):
        return set()

    try:
        old_df = pd.read_csv(output_csv, usecols=[HASH_COL])
        return set(old_df[HASH_COL].astype(str).tolist())
    except Exception:
        return set()


def main():
    print("=" * 80)
    print("Step: Extract ESM-2 embeddings from CD-HIT 40% dataset")
    print("=" * 80)

    device = get_device()
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Torch CUDA: {torch.version.cuda}")

    df = pd.read_csv(INPUT_CSV)
    required_cols = [ID_COL, HASH_COL, TEMP_COL, SEQ_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")

    df = df.dropna(subset=required_cols).copy()
    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()

    finished_hashes = load_finished_hashes(OUTPUT_CSV)
    if finished_hashes:
        df = df[~df[HASH_COL].astype(str).isin(finished_hashes)].copy()
        print(f"[INFO] Resume mode, remaining rows: {len(df)}")

    if len(df) == 0:
        print("[INFO] No rows left to process.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    use_autocast = torch.cuda.is_available() and USE_FP16
    rows_buffer: List[Dict] = []
    start_time = time.time()

    for idx, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Embedding")):
        accession = str(getattr(row, ID_COL))
        seq_hash = str(getattr(row, HASH_COL))
        target_temp = float(getattr(row, TEMP_COL))
        sequence = str(getattr(row, SEQ_COL))

        try:
            inputs = tokenizer(
                sequence,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                add_special_tokens=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                if use_autocast:
                    with torch.amp.autocast("cuda"):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

            cls_vec = outputs.last_hidden_state[0, 0, :].detach().float().cpu().numpy()

            out_row = {
                "UniProt_Accession": accession,
                "Sequence_Hash": seq_hash,
                "Target_Temperature": target_temp,
            }
            for i, val in enumerate(cls_vec):
                out_row[f"ESM_{i}"] = float(val)

            rows_buffer.append(out_row)

        except Exception as e:
            print(f"\n[WARN] Skip row {idx}, accession={accession}, error={e}")
            clear_memory()
            continue
        finally:
            clear_memory()

        if len(rows_buffer) >= SAVE_EVERY:
            chunk_df = pd.DataFrame(rows_buffer)
            if os.path.exists(OUTPUT_CSV):
                chunk_df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
            else:
                chunk_df.to_csv(OUTPUT_CSV, index=False)
            rows_buffer = []

    if rows_buffer:
        chunk_df = pd.DataFrame(rows_buffer)
        if os.path.exists(OUTPUT_CSV):
            chunk_df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
        else:
            chunk_df.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"[DONE] Output: {OUTPUT_CSV}")
    print(f"[INFO] Time elapsed: {elapsed / 60:.2f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()