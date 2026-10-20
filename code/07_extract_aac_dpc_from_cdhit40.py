import hashlib
from collections import Counter
from itertools import product
from typing import List

import pandas as pd

INPUT_CSV = "05_main_dataset_cdhit40.csv"

ID_COL = "UniProt_Accession"
TEMP_COL = "Target_Temperature"
SEQ_COL = "Protein_Sequence"

MIN_LEN = 20

CLEAN_OUT = "06_cdhit40_clean_dataset.csv"
AAC_OUT = "06_cdhit40_aac_features.csv"
DPC_OUT = "06_cdhit40_dpc_features.csv"

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AA_SET = set(STANDARD_AA)
DIPEPTIDES = [a + b for a, b in product(STANDARD_AA, repeat=2)]


def clean_sequence(seq: str) -> str:
    seq = str(seq).strip().upper()
    return "".join([aa for aa in seq if aa in STANDARD_AA_SET])


def has_only_standard(seq: str) -> bool:
    seq = str(seq).strip().upper()
    return len(seq) > 0 and all(aa in STANDARD_AA_SET for aa in seq)


def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()


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
    print("=" * 70)
    print("Step: Extract AAC/DPC from CD-HIT 40% dataset")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Raw rows: {len(df)}")
    print(f"[INFO] Columns: {list(df.columns)}")

    required_cols = [ID_COL, TEMP_COL, SEQ_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少列: {missing_cols}")

    df = df[[ID_COL, TEMP_COL, SEQ_COL]].copy()
    df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors="coerce")
    df = df.dropna(subset=[ID_COL, TEMP_COL, SEQ_COL]).copy()

    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()
    df["Has_Only_Standard_AA"] = df[SEQ_COL].apply(has_only_standard)
    df["Clean_Sequence"] = df[SEQ_COL].apply(clean_sequence)
    df["Sequence_Length"] = df["Clean_Sequence"].str.len()

    non_standard_count = (~df["Has_Only_Standard_AA"]).sum()
    too_short_count = (df["Sequence_Length"] < MIN_LEN).sum()

    print(f"[INFO] Non-standard AA rows: {non_standard_count}")
    print(f"[INFO] Length < {MIN_LEN}: {too_short_count}")

    df = df[df["Has_Only_Standard_AA"]].copy()
    df = df[df["Sequence_Length"] >= MIN_LEN].copy()

    df["Sequence_Hash"] = df["Clean_Sequence"].apply(seq_hash)

    clean_df = pd.DataFrame({
        "UniProt_Accession": df[ID_COL].astype(str),
        "Sequence_Hash": df["Sequence_Hash"],
        "Target_Temperature": df[TEMP_COL].astype(float),
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
            **{f"AAC_{aa}": val for aa, val in zip(STANDARD_AA, aac_vector(seq))}
        })

        dpc_rows.append({
            "UniProt_Accession": row.UniProt_Accession,
            "Sequence_Hash": row.Sequence_Hash,
            "Target_Temperature": row.Target_Temperature,
            **{f"DPC_{dp}": val for dp, val in zip(DIPEPTIDES, dpc_vector(seq))}
        })

    pd.DataFrame(aac_rows).to_csv(AAC_OUT, index=False)
    pd.DataFrame(dpc_rows).to_csv(DPC_OUT, index=False)

    print(f"[INFO] Saved: {AAC_OUT}")
    print(f"[INFO] Saved: {DPC_OUT}")
    print("=" * 70)
    print("[DONE] AAC/DPC extraction finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()