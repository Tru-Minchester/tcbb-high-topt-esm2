from pathlib import Path
from typing import Dict, List
from collections import Counter
import hashlib

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# ===== input =====
RAW_INPUT = BASE_DIR / "18_external_seq2topt_test_raw.csv"
TRAIN_INPUT = BASE_DIR / "13_cdhit40_strict_nomixed_dataset.csv"

# ===== outputs =====
STD_OUT = BASE_DIR / "18_external_standardized.csv"

COLLAPSED_ALL_OUT = BASE_DIR / "19_external_collapsed_all.csv"
COLLAPSED_MAIN_OUT = BASE_DIR / "19_external_collapsed_main.csv"
CONFLICT_OUT = BASE_DIR / "19_external_conflicts.csv"

CLEAN_OUT = BASE_DIR / "20_external_clean.csv"
NO_OVERLAP_OUT = BASE_DIR / "20_external_clean_no_exact_overlap.csv"

TRAIN_FASTA_OUT = BASE_DIR / "13_train_dev_nomixed.fasta"
EXT_FASTA_OUT = BASE_DIR / "20_external_clean_no_exact_overlap.fasta"

SUMMARY_OUT = BASE_DIR / "18_prepare_external_summary.txt"

# ===== config =====
TEMP_THRESHOLD = 60.0
MIN_LEN = 50
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
FASTA_WIDTH = 80


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


def has_only_standard_aa(seq: str) -> bool:
    return len(seq) > 0 and all(aa in STANDARD_AA for aa in seq)


def wrap_sequence(seq: str, width: int = FASTA_WIDTH) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def build_fasta_header(
    seq_hash_value: str,
    label: int,
    temp_median: float,
    accession: str,
    seq_len: int,
    source_db: str,
) -> str:
    accession = str(accession).replace("|", "_").replace(" ", "_")
    source_db = str(source_db).replace("|", "_").replace(" ", "_")
    return (
        f">{seq_hash_value}"
        f"|label={label}"
        f"|temp_median={temp_median:.2f}"
        f"|len={seq_len}"
        f"|acc={accession}"
        f"|src={source_db}"
    )


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
    seq = group["Protein_Sequence"].iloc[0]
    temps = sorted(group["Target_Temperature"].astype(float).tolist())
    label_info = decide_group_label(temps)

    record = {
        "Sequence_Hash": group["Sequence_Hash"].iloc[0],
        "Protein_Sequence": seq,
        "Sequence_Length": len(seq),
        "N_Measurements": len(group),
        "N_Unique_Accessions": group["UniProt_Accession"].astype(str).nunique(),
        "Representative_Accession": sorted(group["UniProt_Accession"].astype(str).unique())[0],
        "All_Accessions": uniq_join(group["UniProt_Accession"]),
        "Temp_Min": min(temps),
        "Temp_Max": max(temps),
        "Temp_Mean": float(pd.Series(temps).mean()),
        "Temp_Median": float(pd.Series(temps).median()),
        "All_Temperatures": "|".join(map(str, temps)),
        "Source_DB": "Seq2Topt_holdout",
        **label_info,
    }
    return record


def export_fasta(df: pd.DataFrame, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            header = build_fasta_header(
                seq_hash_value=row["Sequence_Hash"],
                label=int(row["Binary_Label"]),
                temp_median=float(row["Temp_Median"]),
                accession=row["Representative_Accession"],
                seq_len=int(row["Sequence_Length"]),
                source_db=row.get("Source_DB", "unknown"),
            )
            f.write(header + "\n")
            f.write(wrap_sequence(row["Protein_Sequence"]) + "\n")


def export_train_fasta(train_df: pd.DataFrame, out_path: Path):
    # 主训练集可能没有 Target_Temperature，而是用 Temp_Median
    temp_col = None
    if "Target_Temperature" in train_df.columns:
        temp_col = "Target_Temperature"
    elif "Temp_Median" in train_df.columns:
        temp_col = "Temp_Median"
    else:
        raise ValueError(
            "Train dataset missing both 'Target_Temperature' and 'Temp_Median'"
        )

    required = ["Sequence_Hash", "Protein_Sequence", temp_col]
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise ValueError(f"Train dataset missing columns: {missing}")

    train_df = train_df.copy()

    if "Binary_Label" not in train_df.columns:
        train_df["Binary_Label"] = (
            pd.to_numeric(train_df[temp_col], errors="coerce") >= TEMP_THRESHOLD
        ).astype(int)

    if "Sequence_Length" not in train_df.columns:
        train_df["Sequence_Length"] = train_df["Protein_Sequence"].astype(str).str.len()

    accession_col = "Representative_Accession" if "Representative_Accession" in train_df.columns else None

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in train_df.drop_duplicates(subset=["Sequence_Hash"]).iterrows():
            accession = row[accession_col] if accession_col else "NA"
            temp_value = float(row[temp_col])
            seq = str(row["Protein_Sequence"]).strip().upper()

            header = build_fasta_header(
                seq_hash_value=row["Sequence_Hash"],
                label=int(row["Binary_Label"]),
                temp_median=temp_value,
                accession=accession,
                seq_len=int(len(seq)),
                source_db="train_dev_nomixed",
            )
            f.write(header + "\n")
            f.write(wrap_sequence(seq) + "\n")


def main():
    print("=" * 90)
    print("Step 18: prepare external Seq2Topt holdout set")
    print("=" * 90)
    print(f"[INFO] Raw external input: {RAW_INPUT}")
    print(f"[INFO] Train/dev benchmark input: {TRAIN_INPUT}")

    for p in [RAW_INPUT, TRAIN_INPUT]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # ------------------------------------------------------------------
    # 1) standardize raw external table
    # ------------------------------------------------------------------
    raw_df = pd.read_csv(RAW_INPUT)
    print(f"[INFO] Raw external rows: {len(raw_df)}")
    print(f"[INFO] Raw external columns: {list(raw_df.columns)}")

    required_raw = ["uniprot_id", "topt", "sequence"]
    missing_raw = [c for c in required_raw if c not in raw_df.columns]
    if missing_raw:
        raise ValueError(f"External raw file missing columns: {missing_raw}")

    std_df = pd.DataFrame({
        "UniProt_Accession": raw_df["uniprot_id"].astype(str).str.strip(),
        "Target_Temperature": pd.to_numeric(raw_df["topt"], errors="coerce"),
        "Protein_Sequence": raw_df["sequence"].astype(str).str.strip().str.upper(),
        "Source_DB": "Seq2Topt_holdout",
    }).dropna(subset=["Target_Temperature", "Protein_Sequence"]).copy()

    std_df["Sequence_Hash"] = std_df["Protein_Sequence"].apply(seq_hash)
    std_df["Sequence_Length"] = std_df["Protein_Sequence"].str.len()

    std_df.to_csv(STD_OUT, index=False)
    print(f"[OK] Saved standardized external table: {STD_OUT}")

    # ------------------------------------------------------------------
    # 2) exact-sequence collapse
    # ------------------------------------------------------------------
    collapsed_rows = []
    for _, group in std_df.groupby("Sequence_Hash", sort=False):
        collapsed_rows.append(summarize_group(group))

    collapsed_df = pd.DataFrame(collapsed_rows)
    collapsed_df = collapsed_df.sort_values(
        by=["Conflict_Flag", "Sequence_Length", "N_Measurements"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    collapsed_main_df = collapsed_df[collapsed_df["Conflict_Flag"] == 0].copy()
    conflict_df = collapsed_df[collapsed_df["Conflict_Flag"] == 1].copy()

    collapsed_df.to_csv(COLLAPSED_ALL_OUT, index=False)
    collapsed_main_df.to_csv(COLLAPSED_MAIN_OUT, index=False)
    conflict_df.to_csv(CONFLICT_OUT, index=False)

    print(f"[OK] Saved collapsed all: {COLLAPSED_ALL_OUT}")
    print(f"[OK] Saved collapsed main: {COLLAPSED_MAIN_OUT}")
    print(f"[OK] Saved conflicts: {CONFLICT_OUT}")

    # ------------------------------------------------------------------
    # 3) clean external set
    # ------------------------------------------------------------------
    clean_df = collapsed_main_df.copy()
    clean_df["Has_Only_Standard_AA"] = clean_df["Protein_Sequence"].apply(has_only_standard_aa)
    clean_df = clean_df[clean_df["Has_Only_Standard_AA"]].copy()
    clean_df = clean_df[clean_df["Sequence_Length"] >= MIN_LEN].copy()
    clean_df = clean_df.drop(columns=["Has_Only_Standard_AA"]).reset_index(drop=True)

    clean_df.to_csv(CLEAN_OUT, index=False)
    print(f"[OK] Saved clean external set: {CLEAN_OUT}")

    # ------------------------------------------------------------------
    # 4) remove exact overlaps with train/dev benchmark
    # ------------------------------------------------------------------
    train_df = pd.read_csv(TRAIN_INPUT)
    if "Sequence_Hash" not in train_df.columns:
        raise ValueError("Train/dev benchmark missing Sequence_Hash")

    train_hashes = set(train_df["Sequence_Hash"].astype(str))
    before_overlap = len(clean_df)
    no_overlap_df = clean_df[~clean_df["Sequence_Hash"].astype(str).isin(train_hashes)].copy()
    removed_exact = before_overlap - len(no_overlap_df)

    no_overlap_df.to_csv(NO_OVERLAP_OUT, index=False)
    print(f"[OK] Saved no-exact-overlap external set: {NO_OVERLAP_OUT}")

    # ------------------------------------------------------------------
    # 5) export FASTA for cd-hit-2d
    # ------------------------------------------------------------------
    export_train_fasta(train_df.copy(), TRAIN_FASTA_OUT)
    export_fasta(no_overlap_df, EXT_FASTA_OUT)

    print(f"[OK] Saved train/dev FASTA: {TRAIN_FASTA_OUT}")
    print(f"[OK] Saved external FASTA: {EXT_FASTA_OUT}")

    # ------------------------------------------------------------------
    # 6) summary
    # ------------------------------------------------------------------
    raw_pos = int((std_df["Target_Temperature"] >= TEMP_THRESHOLD).sum())
    raw_neg = int((std_df["Target_Temperature"] < TEMP_THRESHOLD).sum())

    clean_pos = int((clean_df["Binary_Label"] == 1).sum()) if len(clean_df) else 0
    clean_neg = int((clean_df["Binary_Label"] == 0).sum()) if len(clean_df) else 0

    overlap_pos = int((no_overlap_df["Binary_Label"] == 1).sum()) if len(no_overlap_df) else 0
    overlap_neg = int((no_overlap_df["Binary_Label"] == 0).sum()) if len(no_overlap_df) else 0

    summary_lines = [
        "Step 18 summary",
        f"Raw external rows: {len(raw_df)}",
        f"Standardized rows: {len(std_df)}",
        f"Raw positives (>=60C): {raw_pos}",
        f"Raw negatives (<60C): {raw_neg}",
        f"Collapsed unique sequences: {len(collapsed_df)}",
        f"Conflict sequences removed: {len(conflict_df)}",
        f"Clean external rows after AA/length filter: {len(clean_df)}",
        f"Clean positives: {clean_pos}",
        f"Clean negatives: {clean_neg}",
        f"Exact overlaps removed vs train/dev: {removed_exact}",
        f"External rows after exact-overlap removal: {len(no_overlap_df)}",
        f"Remaining positives after exact-overlap removal: {overlap_pos}",
        f"Remaining negatives after exact-overlap removal: {overlap_neg}",
        "",
        "Next command:",
        "cd-hit-2d -i 13_train_dev_nomixed.fasta -i2 20_external_clean_no_exact_overlap.fasta -o 22_external_novel40.fasta -c 0.4 -n 2 -aS 0.8 -aL 0.8 -M 0 -T 0",
    ]
    SUMMARY_OUT.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[OK] Saved summary: {SUMMARY_OUT}")

    print("-" * 90)
    print(f"[INFO] Clean external rows: {len(clean_df)}")
    print(f"[INFO] Exact overlaps removed: {removed_exact}")
    print(f"[INFO] External rows after exact-overlap removal: {len(no_overlap_df)}")
    print(f"[INFO] Remaining positives: {overlap_pos}")
    print(f"[INFO] Remaining negatives: {overlap_neg}")
    print("-" * 90)
    print("[INFO] Next command to run:")
    print("cd-hit-2d -i 13_train_dev_nomixed.fasta -i2 20_external_clean_no_exact_overlap.fasta -o 22_external_novel40.fasta -c 0.4 -n 2 -aS 0.8 -aL 0.8 -M 0 -T 0")
    print("=" * 90)
    print("[DONE] External Seq2Topt preparation finished.")
    print("=" * 90)


if __name__ == "__main__":
    main()