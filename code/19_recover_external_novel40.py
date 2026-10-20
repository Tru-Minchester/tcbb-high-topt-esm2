from pathlib import Path
from typing import List, Dict

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "20_external_clean_no_exact_overlap.csv"
REP_FASTA = BASE_DIR / "22_external_novel40.fasta"

OUT_CSV = BASE_DIR / "23_external_final_dataset.csv"
SUMMARY_TXT = BASE_DIR / "23_recover_external_novel40_summary.txt"


def parse_fasta_headers(fasta_path: Path) -> List[str]:
    headers = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                headers.append(line[1:])
    return headers


def hash_from_header(header: str) -> str:
    return header.split("|")[0].strip()


def parse_header_meta(header: str) -> Dict:
    parts = header.split("|")
    seq_hash = parts[0].strip()
    meta = {"Sequence_Hash": seq_hash}

    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            meta[k] = v

    return meta


def main():
    print("=" * 90)
    print("Step 23: recover external novel40 dataset")
    print("=" * 90)
    print(f"[INFO] Input external clean CSV: {INPUT_CSV}")
    print(f"[INFO] Input novel40 FASTA: {REP_FASTA}")

    for p in [INPUT_CSV, REP_FASTA]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    source_df = pd.read_csv(INPUT_CSV)
    if "Sequence_Hash" not in source_df.columns:
        raise ValueError("Missing Sequence_Hash in external clean CSV")

    source_df = source_df.drop_duplicates(subset=["Sequence_Hash"]).copy()
    source_df["Sequence_Hash"] = source_df["Sequence_Hash"].astype(str)

    headers = parse_fasta_headers(REP_FASTA)
    rep_hashes = [hash_from_header(h) for h in headers]
    rep_hashes_unique = list(dict.fromkeys(rep_hashes))

    rep_df = source_df[source_df["Sequence_Hash"].isin(rep_hashes_unique)].copy()
    rep_df = rep_df.set_index("Sequence_Hash").loc[rep_hashes_unique].reset_index()

    # 从 header 再补一份信息，方便检查
    header_meta_rows = []
    for h in headers:
        meta = parse_header_meta(h)
        row = {
            "Sequence_Hash": meta.get("Sequence_Hash", ""),
            "Header_Label": int(meta["label"]) if "label" in meta else None,
            "Header_Temp_Median": float(meta["temp_median"]) if "temp_median" in meta else None,
            "Header_Length": int(meta["len"]) if "len" in meta else None,
            "Header_Accession": meta.get("acc", ""),
            "Header_Source_DB": meta.get("src", ""),
        }
        header_meta_rows.append(row)

    header_meta_df = pd.DataFrame(header_meta_rows)
    rep_df = rep_df.merge(header_meta_df, on="Sequence_Hash", how="left")

    rep_df.to_csv(OUT_CSV, index=False)

    pos = int((rep_df["Binary_Label"] == 1).sum()) if "Binary_Label" in rep_df.columns else -1
    neg = int((rep_df["Binary_Label"] == 0).sum()) if "Binary_Label" in rep_df.columns else -1
    ratio = (neg / pos) if pos > 0 else float("inf")

    summary_lines = [
        "Step 23 summary",
        f"External clean no-overlap rows: {len(source_df)}",
        f"Novel40 FASTA entries: {len(rep_hashes_unique)}",
        f"Recovered external final rows: {len(rep_df)}",
        f"Positives: {pos}",
        f"Negatives: {neg}",
        f"Positive:negative ratio = 1:{ratio:.2f}" if pos > 0 else "No positive samples",
        f"Saved external final dataset: {OUT_CSV.name}",
    ]
    SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

    print("-" * 90)
    print(f"[OK] Saved external final dataset: {OUT_CSV}")
    print(f"[OK] Saved summary: {SUMMARY_TXT}")
    print("-" * 90)
    print(f"[INFO] Recovered external final rows: {len(rep_df)}")
    print(f"[INFO] Positives: {pos}")
    print(f"[INFO] Negatives: {neg}")
    if pos > 0:
        print(f"[INFO] Positive:negative ratio = 1:{ratio:.2f}")
    print("=" * 90)
    print("[DONE] External novel40 recovery finished.")
    print("=" * 90)


if __name__ == "__main__":
    main()