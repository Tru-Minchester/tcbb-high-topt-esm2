from pathlib import Path
from typing import List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "10_sequence_label_collapsed_main.csv"
CLEAN_CSV = BASE_DIR / "11_unique_clean_sequences.csv"
FASTA_OUT = BASE_DIR / "11_unique_clean_sequences.fasta"
EXCLUDED_CSV = BASE_DIR / "11_excluded_sequences.csv"
SUMMARY_TXT = BASE_DIR / "11_build_strict_cdhit40_summary.txt"

SEQ_COL = "Protein_Sequence"
HASH_COL = "Sequence_Hash"
LABEL_COL = "Binary_Label"
TEMP_MEDIAN_COL = "Temp_Median"
N_MEAS_COL = "N_Measurements"
ACC_COL = "Representative_Accession"
LEN_COL = "Sequence_Length"

MIN_LEN = 50
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
FASTA_WIDTH = 80


def normalize_sequence(seq: str) -> str:
    return str(seq).strip().upper()


def has_only_standard_aa(seq: str) -> bool:
    return all(aa in VALID_AA for aa in seq)


def sanitize_token(value) -> str:
    text = str(value).strip()
    for ch in ["|", " ", ";", ",", "\t", "\n", "\r"]:
        text = text.replace(ch, "_")
    return text


def wrap_sequence(seq: str, width: int = FASTA_WIDTH) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def build_exclusion_reasons(row: pd.Series) -> List[str]:
    reasons = []
    seq = row[SEQ_COL]

    if not isinstance(seq, str) or not seq:
        reasons.append("empty_sequence")
        return reasons

    if len(seq) < MIN_LEN:
        reasons.append(f"too_short_lt_{MIN_LEN}")

    if not has_only_standard_aa(seq):
        reasons.append("contains_nonstandard_aa")

    label = row[LABEL_COL]
    if pd.isna(label):
        reasons.append("missing_binary_label")

    return reasons


def build_fasta_header(row: pd.Series) -> str:
    seq_hash = sanitize_token(row[HASH_COL])
    label = int(row[LABEL_COL])
    temp_median = float(row[TEMP_MEDIAN_COL])
    n_meas = int(row[N_MEAS_COL])
    accession = sanitize_token(row.get(ACC_COL, "NA"))
    seq_len = int(row[LEN_COL])
    return (
        f">{seq_hash}"
        f"|label={label}"
        f"|temp_median={temp_median:.2f}"
        f"|n={n_meas}"
        f"|len={seq_len}"
        f"|acc={accession}"
    )


def main():
    print("=" * 80)
    print("Step 11: build strict pre-CD-HIT FASTA from collapsed exact-sequence table")
    print("=" * 80)
    print(f"[INFO] Working directory: {Path.cwd()}")
    print(f"[INFO] Script directory: {BASE_DIR}")
    print(f"[INFO] Input CSV path: {INPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}\n"
            f"Please run 10_collapse_exact_sequence_labels.py first."
        )

    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Raw rows from collapsed main table: {len(df)}")
    print(f"[INFO] Columns: {list(df.columns)}")

    required_cols = [HASH_COL, SEQ_COL, LABEL_COL, TEMP_MEDIAN_COL, N_MEAS_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[SEQ_COL] = df[SEQ_COL].apply(normalize_sequence)
    df[LEN_COL] = df[SEQ_COL].apply(len)

    exclusion_rows = []
    keep_mask = []
    for _, row in df.iterrows():
        reasons = build_exclusion_reasons(row)
        keep_mask.append(len(reasons) == 0)
        if reasons:
            bad_row = row.to_dict()
            bad_row["Exclusion_Reasons"] = "|".join(reasons)
            exclusion_rows.append(bad_row)

    clean_df = df[pd.Series(keep_mask, index=df.index)].copy()
    excluded_df = pd.DataFrame(exclusion_rows)

    before_dedup = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=[HASH_COL]).copy()
    dedup_removed = before_dedup - len(clean_df)

    clean_df = clean_df.sort_values(
        by=[LABEL_COL, LEN_COL, TEMP_MEDIAN_COL],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    clean_df["FASTA_Header"] = clean_df.apply(build_fasta_header, axis=1)

    clean_df.to_csv(CLEAN_CSV, index=False)
    excluded_df.to_csv(EXCLUDED_CSV, index=False)

    with open(FASTA_OUT, "w", encoding="utf-8") as f:
        for _, row in clean_df.iterrows():
            f.write(row["FASTA_Header"] + "\n")
            f.write(wrap_sequence(row[SEQ_COL]) + "\n")

    pos = int((clean_df[LABEL_COL] == 1).sum())
    neg = int((clean_df[LABEL_COL] == 0).sum())
    ratio = (neg / pos) if pos > 0 else float("inf")
    max_len = int(clean_df[LEN_COL].max()) if len(clean_df) else 0
    min_len = int(clean_df[LEN_COL].min()) if len(clean_df) else 0
    median_len = float(clean_df[LEN_COL].median()) if len(clean_df) else 0.0

    summary_lines = [
        "Step 11 summary",
        f"Input rows: {len(df)}",
        f"Excluded rows: {len(excluded_df)}",
        f"Rows kept before hash dedup: {before_dedup}",
        f"Duplicate hashes removed after filtering: {dedup_removed}",
        f"Final clean rows: {len(clean_df)}",
        f"Positives: {pos}",
        f"Negatives: {neg}",
        f"Positive:negative ratio = 1:{ratio:.2f}" if pos > 0 else "No positive samples",
        f"Sequence length min/median/max = {min_len}/{median_len:.1f}/{max_len}",
        "Recommended strict CD-HIT command:",
        "cd-hit -i 11_unique_clean_sequences.fasta -o 12_cdhit40_strict.fasta -c 0.4 -n 2 -aS 0.8 -aL 0.8 -M 0 -T 0",
    ]
    SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

    print("-" * 80)
    print(f"[OK] Saved clean sequence table: {CLEAN_CSV}")
    print(f"[OK] Saved excluded sequence table: {EXCLUDED_CSV}")
    print(f"[OK] Saved strict CD-HIT input FASTA: {FASTA_OUT}")
    print(f"[OK] Saved summary: {SUMMARY_TXT}")
    print("-" * 80)
    print(f"[INFO] Excluded rows: {len(excluded_df)}")
    if len(excluded_df) > 0:
        print("[INFO] Top exclusion reasons:")
        print(excluded_df["Exclusion_Reasons"].value_counts().head(10).to_string())
    print(f"[INFO] Final clean rows: {len(clean_df)}")
    print(f"[INFO] Positives: {pos}")
    print(f"[INFO] Negatives: {neg}")
    print(f"[INFO] Positive:negative ratio = 1:{ratio:.2f}" if pos > 0 else "[INFO] No positive samples")
    print(f"[INFO] Sequence length min/median/max = {min_len}/{median_len:.1f}/{max_len}")
    print("[INFO] Recommended next command:")
    print("cd-hit -i 11_unique_clean_sequences.fasta -o 12_cdhit40_strict.fasta -c 0.4 -n 2 -aS 0.8 -aL 0.8 -M 0 -T 0")
    print("=" * 80)
    print("[DONE] Strict pre-CD-HIT FASTA build finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
