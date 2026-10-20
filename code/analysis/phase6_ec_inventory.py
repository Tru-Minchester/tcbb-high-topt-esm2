from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
RELEASE = ROOT
DATA = RELEASE / "data_processed"
OUT_CSV = ROOT / "results" / "analysis" / "phase6" / "phase6_ec_inventory.csv"
OUT_NOTES = ROOT / "results" / "analysis" / "phase6" / "phase6_ec_inventory_notes.md"
OUT_MANIFEST = ROOT / "results" / "analysis" / "phase6" / "phase6_ec_inventory_manifest.json"

HASH_COL = "Sequence_Hash"
LABEL_COL = "Binary_Label"

EC_LEVEL1_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}

DATASETS = [
    {
        "Dataset_ID": "internal_strict_nomixed",
        "Dataset_Name": "Internal strict no-mixed benchmark",
        "Expected_N": 5263,
        "File": DATA / "13_cdhit40_strict_nomixed_dataset.csv",
    },
    {
        "Dataset_ID": "strict_novel40_N66",
        "Dataset_Name": "Strict Seq2Topt novel40 external",
        "Expected_N": 66,
        "File": DATA / "23_external_final_dataset.csv",
    },
    {
        "Dataset_ID": "seq2topt_noexact_N136",
        "Dataset_Name": "Seq2Topt no-exact-overlap external",
        "Expected_N": 136,
        "File": DATA / "20_external_clean_no_exact_overlap.csv",
    },
    {
        "Dataset_ID": "public_topt_noexact_N984",
        "Dataset_Name": "Public Topt no-exact-overlap external",
        "Expected_N": 984,
        "File": DATA / "33_public_topt_candidate_noexact.csv",
    },
]


def find_ec_columns(columns: list[str]) -> list[str]:
    hits = []
    for col in columns:
        lowered = col.lower()
        if "ec" in lowered and ("number" in lowered or lowered in {"ec", "ec_number"}):
            hits.append(col)
    return hits


def is_valid_ec(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a", "unknown", "-", "[]"}:
        return False
    return bool(re.search(r"\b[1-7]\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|-)\b", text))


def ec_level1(value: Any) -> str:
    if not is_valid_ec(value):
        return ""
    match = re.search(r"\b([1-7])\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|-)\b", str(value))
    return match.group(1) if match else ""


def summarize_dataset(spec: dict) -> list[dict]:
    path = Path(spec["File"])
    df = pd.read_csv(path)
    n = len(df)
    if LABEL_COL not in df.columns:
        raise ValueError(f"{path} missing {LABEL_COL}")
    y = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    ec_cols = find_ec_columns(list(df.columns))

    primary_ec_col = ""
    if "Representative_EC_Number" in df.columns:
        primary_ec_col = "Representative_EC_Number"
    elif "EC_Number" in df.columns:
        primary_ec_col = "EC_Number"
    elif ec_cols:
        primary_ec_col = ec_cols[0]

    all_ec_col = "All_EC_Number" if "All_EC_Number" in df.columns else ""
    primary_has_ec = df[primary_ec_col].apply(is_valid_ec) if primary_ec_col else pd.Series(False, index=df.index)
    any_has_ec = primary_has_ec.copy()
    if all_ec_col:
        any_has_ec = any_has_ec | df[all_ec_col].apply(is_valid_ec)

    rows = [
        {
            "Row_Type": "dataset_coverage",
            "Dataset_ID": spec["Dataset_ID"],
            "Dataset_Name": spec["Dataset_Name"],
            "Data_File": str(path),
            "Expected_N": spec["Expected_N"],
            "N": n,
            "Positives": pos,
            "Negatives": neg,
            "Positive_Rate": pos / n if n else "",
            "EC_Columns_Found": "|".join(ec_cols),
            "Primary_EC_Field_For_Level1": primary_ec_col,
            "All_EC_Field": all_ec_col,
            "Primary_EC_Covered_N": int(primary_has_ec.sum()),
            "Primary_EC_Coverage": float(primary_has_ec.mean()) if n else "",
            "Any_EC_Covered_N": int(any_has_ec.sum()),
            "Any_EC_Coverage": float(any_has_ec.mean()) if n else "",
            "EC_Level1": "",
            "EC_Level1_Name": "",
            "EC_Level1_N": "",
            "EC_Level1_Positives": "",
            "EC_Level1_Negatives": "",
            "EC_Level1_Positive_Rate": "",
            "EC_Source_Status": "available_in_dataset" if ec_cols else "missing_in_dataset",
            "Notes": "Coverage and distribution use available dataset EC fields only; no UniProt backfill performed.",
        }
    ]

    if primary_ec_col:
        ec1 = df[primary_ec_col].apply(ec_level1)
        covered = df[primary_has_ec].copy()
        covered["_EC_Level1"] = ec1[primary_has_ec]
        covered["_Binary_Label_Int"] = y[primary_has_ec].to_numpy()
        for level in [str(i) for i in range(1, 8)]:
            g = covered[covered["_EC_Level1"].eq(level)]
            n_g = len(g)
            pos_g = int((g["_Binary_Label_Int"] == 1).sum()) if n_g else 0
            neg_g = int((g["_Binary_Label_Int"] == 0).sum()) if n_g else 0
            rows.append(
                {
                    "Row_Type": "ec_level1_distribution",
                    "Dataset_ID": spec["Dataset_ID"],
                    "Dataset_Name": spec["Dataset_Name"],
                    "Data_File": str(path),
                    "Expected_N": spec["Expected_N"],
                    "N": n,
                    "Positives": pos,
                    "Negatives": neg,
                    "Positive_Rate": pos / n if n else "",
                    "EC_Columns_Found": "|".join(ec_cols),
                    "Primary_EC_Field_For_Level1": primary_ec_col,
                    "All_EC_Field": all_ec_col,
                    "Primary_EC_Covered_N": int(primary_has_ec.sum()),
                    "Primary_EC_Coverage": float(primary_has_ec.mean()) if n else "",
                    "Any_EC_Covered_N": int(any_has_ec.sum()),
                    "Any_EC_Coverage": float(any_has_ec.mean()) if n else "",
                    "EC_Level1": level,
                    "EC_Level1_Name": EC_LEVEL1_NAMES[level],
                    "EC_Level1_N": n_g,
                    "EC_Level1_Positives": pos_g,
                    "EC_Level1_Negatives": neg_g,
                    "EC_Level1_Positive_Rate": pos_g / n_g if n_g else "",
                    "EC_Source_Status": "available_in_dataset",
                    "Notes": "Distribution uses primary EC field only to avoid multi-EC double counting.",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_notes(rows: list[dict]) -> None:
    coverage = [r for r in rows if r["Row_Type"] == "dataset_coverage"]
    lines = [
        "# Phase 6 EC Inventory",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This is an inventory only. No EC-stratified performance analysis was run.",
        "",
        "## Coverage",
        "",
    ]
    for r in coverage:
        lines.append(
            f"- {r['Dataset_ID']}: EC columns = `{r['EC_Columns_Found'] or 'none'}`; "
            f"primary coverage = {r['Primary_EC_Covered_N']}/{r['N']} ({float(r['Primary_EC_Coverage']):.1%}); "
            f"any EC coverage = {r['Any_EC_Covered_N']}/{r['N']} ({float(r['Any_EC_Coverage']):.1%})."
        )
    lines += [
        "",
        "## Interpretation Before Analysis",
        "",
        "- The internal strict no-mixed benchmark has EC annotations available in the dataset and can support EC level-1 stratification on the available annotated subset.",
        "- The three external sets do not contain EC fields in the current frozen data files, so their current EC coverage is 0%.",
        "- No UniProt backfill was performed. If external EC stratification is required, it should be a separate approved step with UniProt query date/version recorded, accession mapping logged, no guessed EC labels, and a declared rule for multiple EC numbers.",
        "",
    ]
    OUT_NOTES.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows: list[dict] = []
    for spec in DATASETS:
        rows.extend(summarize_dataset(spec))
    write_csv(OUT_CSV, rows)
    write_notes(rows)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task": "Phase 6 EC inventory only; no stratified performance analysis.",
        "inputs": {spec["Dataset_ID"]: str(spec["File"]) for spec in DATASETS},
        "outputs": {
            "phase6_ec_inventory": str(OUT_CSV),
            "notes": str(OUT_NOTES),
        },
        "ec_assignment_rule": "Use Representative_EC_Number as primary EC field when present; use primary EC field for level-1 distribution to avoid multi-EC double counting.",
        "uniprot_backfill": "not_performed",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
