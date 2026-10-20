from pathlib import Path
from collections import Counter
from itertools import product
from typing import List, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "13_cdhit40_strict_nomixed_dataset.csv"

CLEAN_OUT = BASE_DIR / "14_strict_nomixed_clean_dataset.csv"
AAC_OUT = BASE_DIR / "14_strict_nomixed_aac_features.csv"
DPC_OUT = BASE_DIR / "14_strict_nomixed_dpc_features.csv"

HASH_COL = "Sequence_Hash"
SEQ_COL = "Protein_Sequence"

ACCESSION_CANDIDATES = ["UniProt_Accession", "Representative_Accession"]
TEMP_CANDIDATES = ["Target_Temperature", "Temp_Median"]
LABEL_CANDIDATES = ["Binary_Label"]
LEN_CANDIDATES = ["Sequence_Length"]

TEMP_THRESHOLD = 60.0
MIN_LEN = 50

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AA_SET = set(STANDARD_AA)
DIPEPTIDES = [a + b for a, b in product(STANDARD_AA, repeat=2)]


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"找不到候选列: {candidates}")
    return None


def clean_sequence(seq: str) -> str:
    seq = str(seq).strip().upper()
    return "".join([aa for aa in seq if aa in STANDARD_AA_SET])


def has_only_standard(seq: str) -> bool:
    seq = str(seq).strip().upper()
    return len(seq) > 0 and all(aa in STANDARD_AA_SET for aa in seq)


def aac_vector(seq: str) -> List[float]:
    counts = Counter(seq)
    length = len(seq)
    return [counts[aa] / length for aa in STANDARD_AA]


def dpc_vector(seq: str) -> List[float]:
    if len(seq) < 2:
        return [0.0] * len(DIPEPTIDES)

    pairs = [seq[i:i + 2] for i in range(len(seq) - 1)]
    counts = Counter(pairs)
    denom = len(seq) - 1
    return [counts[dp] / denom for dp in DIPEPTIDES]


def main():
    print("=" * 80)
    print("Step 14: Extract AAC/DPC from strict no-mixed dataset")
    print("=" * 80)
    print(f"[INFO] Input CSV: {INPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"找不到文件: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Raw rows: {len(df)}")
    print(f"[INFO] Columns: {list(df.columns)}")

    acc_col = pick_col(df, ACCESSION_CANDIDATES, required=True)
    temp_col = pick_col(df, TEMP_CANDIDATES, required=True)
    label_col = pick_col(df, LABEL_CANDIDATES, required=False)
    len_col = pick_col(df, LEN_CANDIDATES, required=False)

    required_cols = [acc_col, HASH_COL, temp_col, SEQ_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少列: {missing_cols}")

    keep_cols = [acc_col, HASH_COL, temp_col, SEQ_COL]
    if label_col:
        keep_cols.append(label_col)
    if len_col and len_col not in keep_cols:
        keep_cols.append(len_col)

    df = df[keep_cols].copy()
    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    df = df.dropna(subset=[acc_col, HASH_COL, temp_col, SEQ_COL]).copy()

    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()
    df["Has_Only_Standard_AA"] = df[SEQ_COL].apply(has_only_standard)
    df["Clean_Sequence"] = df[SEQ_COL].apply(clean_sequence)
    df["Sequence_Length"] = df["Clean_Sequence"].str.len()

    non_standard_count = int((~df["Has_Only_Standard_AA"]).sum())
    too_short_count = int((df["Sequence_Length"] < MIN_LEN).sum())

    print(f"[INFO] Non-standard AA rows: {non_standard_count}")
    print(f"[INFO] Length < {MIN_LEN}: {too_short_count}")

    df = df[df["Has_Only_Standard_AA"]].copy()
    df = df[df["Sequence_Length"] >= MIN_LEN].copy()

    if label_col:
        binary_label = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    else:
        binary_label = (df[temp_col].astype(float) >= TEMP_THRESHOLD).astype(int)

    clean_df = pd.DataFrame({
        "UniProt_Accession": df[acc_col].astype(str),
        "Sequence_Hash": df[HASH_COL].astype(str),
        "Target_Temperature": df[temp_col].astype(float),
        "Binary_Label": binary_label.astype(int),
        "Protein_Sequence": df["Clean_Sequence"],
        "Sequence_Length": df["Sequence_Length"].astype(int),
    }).drop_duplicates(subset=["Sequence_Hash"]).copy()

    clean_df.to_csv(CLEAN_OUT, index=False)
    print(f"[INFO] Clean rows retained: {len(clean_df)}")
    print(f"[INFO] Saved: {CLEAN_OUT}")

    aac_rows = []
    dpc_rows = []

    for row in clean_df.itertuples(index=False):
        seq = row.Protein_Sequence

        aac_rows.append({
            "UniProt_Accession": row.UniProt_Accession,
            "Sequence_Hash": row.Sequence_Hash,
            "Target_Temperature": row.Target_Temperature,
            "Binary_Label": row.Binary_Label,
            **{f"AAC_{aa}": val for aa, val in zip(STANDARD_AA, aac_vector(seq))}
        })

        dpc_rows.append({
            "UniProt_Accession": row.UniProt_Accession,
            "Sequence_Hash": row.Sequence_Hash,
            "Target_Temperature": row.Target_Temperature,
            "Binary_Label": row.Binary_Label,
            **{f"DPC_{dp}": val for dp, val in zip(DIPEPTIDES, dpc_vector(seq))}
        })

    pd.DataFrame(aac_rows).to_csv(AAC_OUT, index=False)
    pd.DataFrame(dpc_rows).to_csv(DPC_OUT, index=False)

    print(f"[INFO] Saved: {AAC_OUT}")
    print(f"[INFO] Saved: {DPC_OUT}")
    print("=" * 80)
    print("[DONE] AAC/DPC extraction finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()