from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import json
import time

import pandas as pd
import requests


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PACKAGE_ROOT / "code"
PROJECT_DIR = PACKAGE_ROOT

DATASET = "biomap-research/optimal_temperature"
CONFIG = "default"
SPLITS = ["train", "test"]
PAGE_SIZE = 100

TRAIN_CSV = PROJECT_DIR / "data_processed" / "14_strict_nomixed_clean_dataset.csv"
SEQ2TOPT_NOEXACT_CSV = PROJECT_DIR / "data_processed" / "20_external_clean_no_exact_overlap.csv"

RAW_OUT = PACKAGE_ROOT / "data_processed" / "33_public_topt_candidate_raw.csv"
STD_OUT = PACKAGE_ROOT / "data_processed" / "33_public_topt_candidate_standardized.csv"
NOEXACT_OUT = PACKAGE_ROOT / "data_processed" / "33_public_topt_candidate_noexact.csv"
SUMMARY_OUT = PACKAGE_ROOT / "data_processed" / "33_public_topt_candidate_summary.txt"

TEMP_THRESHOLD = 60.0
MIN_LEN = 50
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

SEQUENCE_CANDIDATES = [
    "sequence",
    "seq",
    "protein_sequence",
    "Protein_Sequence",
    "Sequence",
    "protein",
]
TEMP_CANDIDATES = [
    "label",
    "topt",
    "Topt",
    "optimal_temperature",
    "Target_Temperature",
    "temperature",
    "Temperature",
]


def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()


def normalize_sequence(seq: str) -> str:
    return str(seq).strip().upper()


def has_only_standard_aa(seq: str) -> bool:
    return len(seq) > 0 and all(aa in STANDARD_AA for aa in seq)


def rows_url(split: str, offset: int, length: int) -> str:
    dataset_q = DATASET.replace("/", "%2F")
    return (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={dataset_q}&config={CONFIG}&split={split}&offset={offset}&length={length}"
    )


def fetch_split(split: str) -> List[Dict]:
    rows: List[Dict] = []
    offset = 0
    while True:
        url = rows_url(split, offset, PAGE_SIZE)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        page = payload.get("rows", [])
        if not page:
            break

        for item in page:
            row = item.get("row", {})
            row = dict(row)
            row["HF_Split"] = split
            row["HF_Row_Idx"] = item.get("row_idx")
            rows.append(row)

        offset += len(page)
        total = payload.get("num_rows_total")
        print(f"[INFO] {split}: fetched {len(rows)} / {total if total is not None else '?'} rows")
        if total is not None and offset >= int(total):
            break
        if len(page) < PAGE_SIZE:
            break
        time.sleep(0.2)
    return rows


def pick_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def decide_label(temps: List[float]) -> Dict:
    below = [t for t in temps if t < TEMP_THRESHOLD]
    above = [t for t in temps if t >= TEMP_THRESHOLD]
    if below and above:
        return {
            "Binary_Label": None,
            "Conflict_Flag": 1,
            "Conflict_Type": "cross_threshold",
        }
    if above:
        return {
            "Binary_Label": 1,
            "Conflict_Flag": 0,
            "Conflict_Type": "none",
        }
    return {
        "Binary_Label": 0,
        "Conflict_Flag": 0,
        "Conflict_Type": "none",
    }


def summarize_sequence_group(group: pd.DataFrame) -> Dict:
    seq = group["Protein_Sequence"].iloc[0]
    temps = sorted(group["Target_Temperature"].astype(float).tolist())
    label_info = decide_label(temps)
    return {
        "Sequence_Hash": group["Sequence_Hash"].iloc[0],
        "Protein_Sequence": seq,
        "Sequence_Length": len(seq),
        "N_Measurements": len(group),
        "Temp_Min": min(temps),
        "Temp_Max": max(temps),
        "Temp_Mean": float(pd.Series(temps).mean()),
        "Temp_Median": float(pd.Series(temps).median()),
        "All_Temperatures": "|".join(map(str, temps)),
        "Source_DB": DATASET,
        "Source_Splits": "|".join(sorted(group["HF_Split"].astype(str).unique())),
        **label_info,
    }


def load_overlap_hashes(path: Path, seq_col: str = "Protein_Sequence") -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=[seq_col])
    seqs = df[seq_col].dropna().astype(str).map(normalize_sequence)
    return {seq_hash(seq) for seq in seqs}


def main():
    print("=" * 90)
    print("Step 33: download public Topt candidate set from Hugging Face")
    print("=" * 90)
    print(f"[INFO] Dataset: {DATASET}")
    print(f"[INFO] Output dir: {BASE_DIR}")

    all_rows: List[Dict] = []
    for split in SPLITS:
        all_rows.extend(fetch_split(split))

    if not all_rows:
        raise RuntimeError("No rows downloaded.")

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(RAW_OUT, index=False)
    print(f"[OK] Saved raw rows: {RAW_OUT}")

    seq_col = pick_col(raw_df.columns.tolist(), SEQUENCE_CANDIDATES)
    temp_col = pick_col(raw_df.columns.tolist(), TEMP_CANDIDATES)
    if not seq_col or not temp_col:
        raise ValueError(
            f"Could not identify sequence/temp columns. Columns={raw_df.columns.tolist()}"
        )

    std_df = pd.DataFrame({
        "Source_DB": DATASET,
        "HF_Split": raw_df["HF_Split"],
        "HF_Row_Idx": raw_df["HF_Row_Idx"],
        "Protein_Sequence": raw_df[seq_col].map(normalize_sequence),
        "Target_Temperature": pd.to_numeric(raw_df[temp_col], errors="coerce"),
    })
    std_df = std_df.dropna(subset=["Protein_Sequence", "Target_Temperature"]).copy()
    std_df["Sequence_Length"] = std_df["Protein_Sequence"].str.len()
    std_df["Has_Only_Standard_AA"] = std_df["Protein_Sequence"].apply(has_only_standard_aa)
    std_df = std_df[std_df["Has_Only_Standard_AA"]].copy()
    std_df = std_df[std_df["Sequence_Length"] >= MIN_LEN].copy()
    std_df["Sequence_Hash"] = std_df["Protein_Sequence"].apply(seq_hash)
    std_df["Raw_Binary_Label"] = (std_df["Target_Temperature"] >= TEMP_THRESHOLD).astype(int)
    std_df.to_csv(STD_OUT, index=False)
    print(f"[OK] Saved standardized rows: {STD_OUT}")

    collapsed_rows = [
        summarize_sequence_group(group)
        for _, group in std_df.groupby("Sequence_Hash", sort=False)
    ]
    collapsed_df = pd.DataFrame(collapsed_rows)
    conflict_count = int((collapsed_df["Conflict_Flag"] == 1).sum())
    main_df = collapsed_df[collapsed_df["Conflict_Flag"] == 0].copy()

    train_hashes = load_overlap_hashes(TRAIN_CSV)
    seq2topt_hashes = load_overlap_hashes(SEQ2TOPT_NOEXACT_CSV)
    blocked_hashes = train_hashes | seq2topt_hashes

    noexact_df = main_df[~main_df["Sequence_Hash"].isin(blocked_hashes)].copy()
    noexact_df = noexact_df.reset_index(drop=True)
    noexact_df.to_csv(NOEXACT_OUT, index=False)
    print(f"[OK] Saved no-exact public candidate rows: {NOEXACT_OUT}")

    pos = int((noexact_df["Binary_Label"] == 1).sum()) if len(noexact_df) else 0
    neg = int((noexact_df["Binary_Label"] == 0).sum()) if len(noexact_df) else 0
    summary = [
        "Step 33 summary",
        f"Dataset: {DATASET}",
        f"Raw rows downloaded: {len(raw_df)}",
        f"Identified sequence column: {seq_col}",
        f"Identified temperature column: {temp_col}",
        f"Standardized clean rows: {len(std_df)}",
        f"Collapsed unique sequences: {len(collapsed_df)}",
        f"Cross-threshold conflicts removed: {conflict_count}",
        f"No-exact rows after removing strict-train and Seq2Topt-noexact overlaps: {len(noexact_df)}",
        f"No-exact positives: {pos}",
        f"No-exact negatives: {neg}",
    ]
    SUMMARY_OUT.write_text("\n".join(summary), encoding="utf-8")
    print("-" * 90)
    print("\n".join(summary))
    print("=" * 90)
    print("[DONE] Public Topt candidate download/standardization finished.")
    print("=" * 90)


if __name__ == "__main__":
    main()
