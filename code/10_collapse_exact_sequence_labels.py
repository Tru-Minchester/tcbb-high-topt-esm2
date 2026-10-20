import hashlib
from pathlib import Path
from typing import Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "02_main_dataset_with_sequences.csv"
ALL_OUT = BASE_DIR / "10_sequence_label_collapsed_all.csv"
MAIN_OUT = BASE_DIR / "10_sequence_label_collapsed_main.csv"
CONFLICT_OUT = BASE_DIR / "10_sequence_label_conflicts.csv"

TEMP_THRESHOLD = 60.0
ID_COL = "UniProt_Accession"
TEMP_COL = "Target_Temperature"
SEQ_COL = "Protein_Sequence"

OPTIONAL_COLS = ["EC_Number", "Organism", "Source_DB"]


def normalize_sequence(seq: str) -> str:
    return str(seq).strip().upper()


def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()


def uniq_join(values: pd.Series, sep: str = "|") -> str:
    vals = []
    for v in values.dropna().astype(str):
        v = v.strip()
        if v and v not in vals:
            vals.append(v)
    return sep.join(vals)


def decide_group_label(temps: List[float], threshold: float = TEMP_THRESHOLD) -> Dict:
    below = [t for t in temps if t < threshold]
    above = [t for t in temps if t >= threshold]

    if below and above:
        return {
            "Binary_Label": None,
            "Conflict_Flag": 1,
            "Conflict_Type": "cross_threshold",
            "Label_Status": "exclude_from_main",
        }
    if above:
        return {
            "Binary_Label": 1,
            "Conflict_Flag": 0,
            "Conflict_Type": "none",
            "Label_Status": "all_hot",
        }
    return {
        "Binary_Label": 0,
        "Conflict_Flag": 0,
        "Conflict_Type": "none",
        "Label_Status": "all_cold",
    }


def summarize_group(group: pd.DataFrame) -> Dict:
    seq = group["Normalized_Sequence"].iloc[0]
    temps = sorted(group[TEMP_COL].astype(float).tolist())
    label_info = decide_group_label(temps)

    record = {
        "Sequence_Hash": group["Sequence_Hash"].iloc[0],
        "Protein_Sequence": seq,
        "Sequence_Length": len(seq),
        "N_Measurements": len(group),
        "N_Unique_Accessions": group[ID_COL].astype(str).nunique(),
        "Representative_Accession": sorted(group[ID_COL].astype(str).unique())[0],
        "All_Accessions": uniq_join(group[ID_COL]),
        "Temp_Min": min(temps),
        "Temp_Max": max(temps),
        "Temp_Mean": float(pd.Series(temps).mean()),
        "Temp_Median": float(pd.Series(temps).median()),
        "All_Temperatures": "|".join(map(str, temps)),
        **label_info,
    }

    for col in OPTIONAL_COLS:
        if col in group.columns:
            record[f"All_{col}"] = uniq_join(group[col])
            first_valid = group[col].dropna().astype(str)
            record[f"Representative_{col}"] = first_valid.iloc[0].strip() if len(first_valid) else ""

    return record


def main():
    print("=" * 80)
    print("Step 10: collapse exact-sequence labels before CD-HIT")
    print("=" * 80)
    print(f"[INFO] Working directory: {Path.cwd()}")
    print(f"[INFO] Script directory: {BASE_DIR}")
    print(f"[INFO] Input CSV path: {INPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}\n"
            f"Please put 02_main_dataset_with_sequences.csv in the same folder as this script."
        )

    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Raw rows: {len(df)}")
    print(f"[INFO] Columns: {list(df.columns)}")

    required_cols = [ID_COL, TEMP_COL, SEQ_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors="coerce")
    df[SEQ_COL] = df[SEQ_COL].astype(str)
    df = df.dropna(subset=[ID_COL, TEMP_COL, SEQ_COL]).copy()

    df["Normalized_Sequence"] = df[SEQ_COL].apply(normalize_sequence)
    df = df[df["Normalized_Sequence"] != ""].copy()
    df["Sequence_Hash"] = df["Normalized_Sequence"].apply(seq_hash)

    print(f"[INFO] Rows after basic NA/empty cleanup: {len(df)}")
    print(f"[INFO] Unique exact sequences before collapse: {df['Sequence_Hash'].nunique()}")

    collapsed_rows = []
    for _, group in df.groupby("Sequence_Hash", sort=False):
        collapsed_rows.append(summarize_group(group))

    collapsed_df = pd.DataFrame(collapsed_rows)
    collapsed_df = collapsed_df.sort_values(
        by=["Conflict_Flag", "Sequence_Length", "N_Measurements"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    main_df = collapsed_df[collapsed_df["Conflict_Flag"] == 0].copy()
    conflict_df = collapsed_df[collapsed_df["Conflict_Flag"] == 1].copy()

    collapsed_df.to_csv(ALL_OUT, index=False)
    main_df.to_csv(MAIN_OUT, index=False)
    conflict_df.to_csv(CONFLICT_OUT, index=False)

    print("-" * 80)
    print(f"[OK] Saved all collapsed sequences: {ALL_OUT}")
    print(f"[OK] Saved main benchmark candidates: {MAIN_OUT}")
    print(f"[OK] Saved exact-sequence conflicts: {CONFLICT_OUT}")
    print("-" * 80)
    print(f"[INFO] Total collapsed unique sequences: {len(collapsed_df)}")
    print(f"[INFO] Conflict sequences removed from main: {len(conflict_df)}")
    print(f"[INFO] Main non-conflict sequences: {len(main_df)}")

    if len(main_df) > 0:
        pos = int((main_df["Binary_Label"] == 1).sum())
        neg = int((main_df["Binary_Label"] == 0).sum())
        ratio = (neg / pos) if pos > 0 else float("inf")
        print(f"[INFO] Main positives (>= {TEMP_THRESHOLD}C): {pos}")
        print(f"[INFO] Main negatives (< {TEMP_THRESHOLD}C): {neg}")
        print(f"[INFO] Main positive:negative ratio = 1:{ratio:.2f}" if pos > 0 else "[INFO] No positive samples")

    if len(conflict_df) > 0:
        print("[INFO] Example conflicts:")
        show_cols = [
            "Representative_Accession",
            "Temp_Min",
            "Temp_Max",
            "N_Measurements",
            "All_Temperatures",
        ]
        print(conflict_df[show_cols].head(10).to_string(index=False))

    print("=" * 80)
    print("[DONE] Exact-sequence label collapsing finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
