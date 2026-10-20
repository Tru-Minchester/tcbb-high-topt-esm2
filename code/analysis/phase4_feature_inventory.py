from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
RELEASE = ROOT
FEATURES = RELEASE / "features"
DATA = RELEASE / "data_processed"
RESULTS = RELEASE / "results"
CODE = RELEASE / "code"
OUT_CSV = ROOT / "phase4_feature_inventory.csv"
OUT_NOTES = ROOT / "phase4_feature_inventory_notes.md"

HASH_COL = "Sequence_Hash"
LABEL_COL = "Binary_Label"


DATASETS = [
    {
        "dataset_id": "internal_strict_nomixed",
        "dataset_name": "Internal strict no-mixed benchmark",
        "expected_n": 5263,
        "source_data": DATA / "13_cdhit40_strict_nomixed_dataset.csv",
    },
    {
        "dataset_id": "strict_novel40_N66",
        "dataset_name": "Strict Seq2Topt novel40 external",
        "expected_n": 66,
        "source_data": DATA / "23_external_final_dataset.csv",
    },
    {
        "dataset_id": "seq2topt_noexact_N136",
        "dataset_name": "Seq2Topt no-exact-overlap external",
        "expected_n": 136,
        "source_data": DATA / "20_external_clean_no_exact_overlap.csv",
    },
    {
        "dataset_id": "public_topt_noexact_N984",
        "dataset_name": "Public Topt no-exact-overlap external",
        "expected_n": 984,
        "source_data": DATA / "33_public_topt_candidate_noexact.csv",
    },
]

FEATURE_SPECS = [
    {
        "feature_group": "AAC",
        "expected_dim": 20,
        "prefix": "AAC_",
        "files": {
            "internal_strict_nomixed": FEATURES / "14_strict_nomixed_aac_features.csv",
        },
    },
    {
        "feature_group": "DPC",
        "expected_dim": 400,
        "prefix": "DPC_",
        "files": {
            "internal_strict_nomixed": FEATURES / "14_strict_nomixed_dpc_features.csv",
        },
    },
    {
        "feature_group": "ESM2_150M",
        "expected_dim": 640,
        "prefix": "ESM_",
        "files": {
            "internal_strict_nomixed": FEATURES / "15_strict_nomixed_esm2_features.csv",
            "strict_novel40_N66": FEATURES / "24_external_esm2_features.csv",
            "seq2topt_noexact_N136": FEATURES / "32_external_noexact_esm2_150m_features.csv",
            "public_topt_noexact_N984": FEATURES / "34_public_topt_noexact_esm2_150m_features.csv",
        },
    },
    {
        "feature_group": "ESM2_650M",
        "expected_dim": 1280,
        "prefix": "ESM650_",
        "files": {
            "internal_strict_nomixed": FEATURES / "27_strict_nomixed_esm2_650m_features.csv",
            "strict_novel40_N66": FEATURES / "28_external_esm2_650m_features.csv",
            "seq2topt_noexact_N136": FEATURES / "32_external_noexact_esm2_650m_features.csv",
            "public_topt_noexact_N984": FEATURES / "34_public_topt_noexact_esm2_650m_features.csv",
        },
    },
]


def csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def count_rows_and_labels(path: Path) -> tuple[int, int | None, int | None]:
    usecols = []
    header = csv_header(path)
    for col in [HASH_COL, LABEL_COL]:
        if col in header:
            usecols.append(col)
    if not usecols:
        n = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
        return n, None, None
    df = pd.read_csv(path, usecols=usecols)
    n = len(df)
    if LABEL_COL in df.columns:
        y = pd.to_numeric(df[LABEL_COL], errors="coerce")
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        return n, pos, neg
    return n, None, None


def source_data_summary(path: Path) -> dict:
    if not path.exists():
        return {"source_data_exists": False}
    header = csv_header(path)
    n, pos, neg = count_rows_and_labels(path)
    seq_cols = [c for c in ["Protein_Sequence", "Sequence", "sequence"] if c in header]
    return {
        "source_data_exists": True,
        "source_data_rows": n,
        "source_data_pos": pos,
        "source_data_neg": neg,
        "source_has_sequence": bool(seq_cols),
        "source_sequence_column": seq_cols[0] if seq_cols else "",
    }


def inventory_rows() -> list[dict]:
    rows = []
    data_by_id = {d["dataset_id"]: d for d in DATASETS}
    for dataset in DATASETS:
        source_info = source_data_summary(dataset["source_data"])
        for spec in FEATURE_SPECS:
            path = spec["files"].get(dataset["dataset_id"])
            exists = bool(path and path.exists())
            row = {
                "Dataset_ID": dataset["dataset_id"],
                "Dataset_Name": dataset["dataset_name"],
                "Expected_N": dataset["expected_n"],
                "Feature_Group": spec["feature_group"],
                "Expected_Dim": spec["expected_dim"],
                "Feature_Prefix": spec["prefix"],
                "Feature_File": str(path) if path else "",
                "Feature_File_Exists": exists,
                "Source_Data_File": str(dataset["source_data"]),
                **source_info,
            }
            if exists and path is not None:
                header = csv_header(path)
                feat_cols = [c for c in header if c.startswith(spec["prefix"])]
                n, pos, neg = count_rows_and_labels(path)
                row.update(
                    {
                        "Observed_N": n,
                        "Observed_Pos": pos,
                        "Observed_Neg": neg,
                        "Observed_Dim": len(feat_cols),
                        "Dimension_Matches_Expected": len(feat_cols) == spec["expected_dim"],
                        "Sample_Count_Matches_Expected": n == dataset["expected_n"],
                        "Status": "exists_ready_as_feature_matrix"
                        if len(feat_cols) == spec["expected_dim"] and n == dataset["expected_n"]
                        else "exists_needs_validation",
                        "Needed_Action": "none_for_feature_matrix",
                    }
                )
            else:
                can_recompute = bool(source_info.get("source_data_exists") and source_info.get("source_has_sequence"))
                row.update(
                    {
                        "Observed_N": "",
                        "Observed_Pos": "",
                        "Observed_Neg": "",
                        "Observed_Dim": "",
                        "Dimension_Matches_Expected": "",
                        "Sample_Count_Matches_Expected": "",
                        "Status": "missing_feature_matrix",
                        "Needed_Action": "recompute_from_sequence" if can_recompute else "blocked_missing_sequence_source",
                    }
                )
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_notes(rows: list[dict]) -> None:
    ready = [r for r in rows if r["Status"] == "exists_ready_as_feature_matrix"]
    missing = [r for r in rows if r["Status"] == "missing_feature_matrix"]
    lines = [
        "# Phase 4 Feature Inventory",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Feature Matrix Availability",
        "",
        f"- Ready feature matrices: {len(ready)}/16.",
        f"- Missing feature matrices: {len(missing)}/16.",
        "- ESM2-150M and ESM2-650M matrices exist for the internal strict no-mixed benchmark and all three external sets.",
        "- AAC and DPC matrices exist for the internal strict no-mixed benchmark only.",
        "- AAC and DPC matrices need to be recomputed for N=66, N=136, and N=984 from the sequence-bearing source data before external handcrafted or fusion ablations can be run.",
        "",
        "## Existing Result Reuse Assessment",
        "",
        "Existing results should be treated as historical references, not as directly reusable Phase 4 ablation results, because Phase 4 requires the only variable to be the feature group.",
        "",
        "- `16_strict_nomixed_feature_set_fold_results.csv` contains AAC/DPC/ESM2 internal comparisons, but those runs used nested grid search with per-fold best RF hyperparameters. They do not satisfy the Phase 4 requirement that RF hyperparameters be identical across all branches.",
        "- `29_model_scale_internal_fold_results.csv` and the external ESM2 result files provide frozen ESM2-150M/650M references, but ESM2-150M and ESM2-650M used different RF hyperparameters for the final external models, especially `min_samples_split=2` for 150M and `min_samples_split=5` for 650M.",
        "- Phase 3 calibration reproduced the frozen Phase 29 internal folds with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for the base model and used `CALIBRATION_SEED=20260713` only for calibration-related estimators. Therefore the implemented Phase 3 base-model split is seed 42, not 20260713.",
        "- Existing pipelines consistently use imputer -> scaler -> SMOTE -> RandomForest and fixed threshold 0.5 for thresholded metrics, but the split seed and RF hyperparameters are not uniformly frozen across all historical branches.",
        "",
        "## Recommendation Before Running Phase 4",
        "",
        "For a strict TCBB feature-fusion ablation, rerun all Phase 4 branches under one newly declared manifest rather than reusing historical metrics. The common frozen resources should be explicitly chosen before running:",
        "",
        "- CV split: either use the actual Phase 3/Phase 29-compatible base split seed 42, or deliberately switch to seed 20260713 and rerun every branch from scratch. Do not mix the two.",
        "- RF hyperparameters: choose one common RF parameter set for all branches and record it in the Phase 4 manifest.",
        "- SMOTE strategy: keep `SMOTE(random_state=42)` if preserving historical model behavior, or change all branches together if a new seed is chosen.",
        "- Threshold: keep fixed 0.5 across all threshold-dependent metrics.",
        "",
        "My recommendation is to use the actual frozen Phase 3-compatible base split seed 42 if the priority is continuity with frozen Phase 1-3 results. If the priority is matching the earlier design text that mentioned seed 20260713, then all Phase 4 branches should be rerun under seed 20260713 and clearly documented as a new Phase 4 ablation protocol.",
        "",
        "## Missing Feature Matrices",
        "",
    ]
    for r in missing:
        lines.append(
            f"- {r['Dataset_ID']} / {r['Feature_Group']}: {r['Needed_Action']} from `{r['Source_Data_File']}`."
        )
    OUT_NOTES.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = inventory_rows()
    write_csv(OUT_CSV, rows)
    write_notes(rows)
    print(
        json.dumps(
            {
                "feature_inventory": str(OUT_CSV),
                "notes": str(OUT_NOTES),
                "rows": len(rows),
                "ready": sum(1 for r in rows if r["Status"] == "exists_ready_as_feature_matrix"),
                "missing": sum(1 for r in rows if r["Status"] == "missing_feature_matrix"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
