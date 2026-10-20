from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
RELEASE = ROOT
DATA = RELEASE / "data_processed"
FEATURES = RELEASE / "features"
RESULTS = RELEASE / "results"
OUT = ROOT / "results" / "analysis" / "phase4_feature_matrices"
ALIGNMENT_CSV = OUT / "phase4_external_handcrafted_alignment_checks.csv"
MANIFEST_JSON = OUT / "phase4_external_handcrafted_manifest.json"

HASH_COL = "Sequence_Hash"
SEQ_COL = "Protein_Sequence"
LABEL_COL = "Binary_Label"
TEMP_THRESHOLD = 60.0
MIN_LEN = 50

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AA_SET = set(STANDARD_AA)
DIPEPTIDES = [a + b for a, b in product(STANDARD_AA, repeat=2)]


@dataclass(frozen=True)
class ExternalSet:
    dataset_id: str
    dataset_name: str
    expected_n: int
    source_data: Path
    esm2_150m_features: Path
    esm2_650m_features: Path
    frozen_predictions: Path
    output_prefix: str


EXTERNAL_SETS = [
    ExternalSet(
        dataset_id="strict_novel40_N66",
        dataset_name="Strict Seq2Topt novel40 external",
        expected_n=66,
        source_data=DATA / "23_external_final_dataset.csv",
        esm2_150m_features=FEATURES / "24_external_esm2_features.csv",
        esm2_650m_features=FEATURES / "28_external_esm2_650m_features.csv",
        frozen_predictions=RESULTS / "29_model_scale_external_results.csv",
        output_prefix="strict_novel40_N66",
    ),
    ExternalSet(
        dataset_id="seq2topt_noexact_N136",
        dataset_name="Seq2Topt no-exact-overlap external",
        expected_n=136,
        source_data=DATA / "20_external_clean_no_exact_overlap.csv",
        esm2_150m_features=FEATURES / "32_external_noexact_esm2_150m_features.csv",
        esm2_650m_features=FEATURES / "32_external_noexact_esm2_650m_features.csv",
        frozen_predictions=RESULTS / "32_external_noexact_model_scale_results.csv",
        output_prefix="seq2topt_noexact_N136",
    ),
    ExternalSet(
        dataset_id="public_topt_noexact_N984",
        dataset_name="Public Topt no-exact-overlap external",
        expected_n=984,
        source_data=DATA / "33_public_topt_candidate_noexact.csv",
        esm2_150m_features=FEATURES / "34_public_topt_noexact_esm2_150m_features.csv",
        esm2_650m_features=FEATURES / "34_public_topt_noexact_esm2_650m_features.csv",
        frozen_predictions=RESULTS / "34_public_topt_noexact_model_scale_results.csv",
        output_prefix="public_topt_noexact_N984",
    ),
]


def clean_sequence(seq: str) -> str:
    seq = str(seq).strip().upper()
    return "".join(aa for aa in seq if aa in STANDARD_AA_SET)


def has_only_standard(seq: str) -> bool:
    seq = str(seq).strip().upper()
    return len(seq) > 0 and all(aa in STANDARD_AA_SET for aa in seq)


def aac_vector(seq: str) -> list[float]:
    counts = Counter(seq)
    length = len(seq)
    return [counts[aa] / length for aa in STANDARD_AA]


def dpc_vector(seq: str) -> list[float]:
    if len(seq) < 2:
        return [0.0] * len(DIPEPTIDES)
    pairs = [seq[i : i + 2] for i in range(len(seq) - 1)]
    counts = Counter(pairs)
    denom = len(seq) - 1
    return [counts[dp] / denom for dp in DIPEPTIDES]


def pick_first_existing(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Missing required column from candidates: {candidates}")
    return ""


def load_source(spec: ExternalSet, target_hash_order: list[str]) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(spec.source_data)
    acc_col = pick_first_existing(df, ["UniProt_Accession", "Representative_Accession"], required=False)
    temp_col = pick_first_existing(df, ["Target_Temperature", "Temp_Median", "Temp_Mean"], required=True)
    if HASH_COL not in df.columns or SEQ_COL not in df.columns:
        raise ValueError(f"{spec.source_data} must contain {HASH_COL} and {SEQ_COL}")

    source_raw_n = len(df)
    duplicate_hashes = int(df[HASH_COL].duplicated().sum())
    df = df.dropna(subset=[HASH_COL, SEQ_COL, temp_col]).copy()
    df[HASH_COL] = df[HASH_COL].astype(str)
    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()
    df["Has_Only_Standard_AA"] = df[SEQ_COL].apply(has_only_standard)
    df["Clean_Sequence"] = df[SEQ_COL].apply(clean_sequence)
    df["Sequence_Length_Clean"] = df["Clean_Sequence"].str.len()

    nonstandard = int((~df["Has_Only_Standard_AA"]).sum())
    too_short = int((df["Sequence_Length_Clean"] < MIN_LEN).sum())

    if LABEL_COL in df.columns:
        y = pd.to_numeric(df[LABEL_COL], errors="coerce").astype(int)
    else:
        y = (pd.to_numeric(df[temp_col], errors="coerce").astype(float) >= TEMP_THRESHOLD).astype(int)

    clean = pd.DataFrame(
        {
            "UniProt_Accession": df[acc_col].astype(str) if acc_col else "NA",
            HASH_COL: df[HASH_COL].astype(str),
            "Target_Temperature": pd.to_numeric(df[temp_col], errors="coerce").astype(float),
            "Binary_Label": y.astype(int),
            "Protein_Sequence": df["Clean_Sequence"],
            "Sequence_Length": df["Sequence_Length_Clean"].astype(int),
            "Has_Only_Standard_AA": df["Has_Only_Standard_AA"].astype(bool),
        }
    ).drop_duplicates(subset=[HASH_COL])

    missing_from_source = [h for h in target_hash_order if h not in set(clean[HASH_COL])]
    extra_in_source = [h for h in clean[HASH_COL].astype(str) if h not in set(target_hash_order)]
    if missing_from_source or extra_in_source:
        raise ValueError(
            f"{spec.dataset_id}: source/ESM2 hash set mismatch; missing={len(missing_from_source)}, extra={len(extra_in_source)}"
        )

    clean = clean.set_index(HASH_COL).loc[target_hash_order].reset_index()
    diagnostics = {
        "Source_Raw_N": source_raw_n,
        "Source_Deduplicated_N": len(clean),
        "Source_Duplicate_Hashes": duplicate_hashes,
        "Nonstandard_Sequence_Count": nonstandard,
        "Clean_Length_LT_50_Count": too_short,
    }
    return clean, diagnostics


def make_feature_tables(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aac_rows = []
    dpc_rows = []
    for row in clean.itertuples(index=False):
        seq = row.Protein_Sequence
        base = {
            "UniProt_Accession": row.UniProt_Accession,
            "Sequence_Hash": row.Sequence_Hash,
            "Target_Temperature": float(row.Target_Temperature),
            "Binary_Label": int(row.Binary_Label),
        }
        aac_rows.append({**base, **{f"AAC_{aa}": val for aa, val in zip(STANDARD_AA, aac_vector(seq))}})
        dpc_rows.append({**base, **{f"DPC_{dp}": val for dp, val in zip(DIPEPTIDES, dpc_vector(seq))}})
    return pd.DataFrame(aac_rows), pd.DataFrame(dpc_rows)


def feature_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def compare_hash_label(
    spec: ExternalSet,
    feature_name: str,
    feature_df: pd.DataFrame,
    compare_name: str,
    compare_df: pd.DataFrame,
    compare_label_col: str,
    diagnostics: dict,
) -> dict:
    n = len(feature_df)
    hash_mismatch = int((feature_df[HASH_COL].astype(str).to_numpy() != compare_df[HASH_COL].astype(str).to_numpy()).sum())
    label_mismatch = int((feature_df["Binary_Label"].astype(int).to_numpy() != compare_df[compare_label_col].astype(int).to_numpy()).sum())
    return {
        "Dataset_ID": spec.dataset_id,
        "Dataset_Name": spec.dataset_name,
        "Feature_Group": feature_name,
        "Compared_To": compare_name,
        "N_Feature": n,
        "N_Compared": len(compare_df),
        "Expected_N": spec.expected_n,
        "Feature_Dim": len(feature_cols(feature_df, "AAC_" if feature_name == "AAC" else "DPC_")),
        "Hash_Order_Mismatch": hash_mismatch,
        "Label_Mismatch": label_mismatch,
        "N_Matches_Expected": n == spec.expected_n and len(compare_df) == spec.expected_n,
        "Mismatch_OK": hash_mismatch == 0 and label_mismatch == 0 and n == len(compare_df) == spec.expected_n,
        **diagnostics,
    }


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


def process_one(spec: ExternalSet) -> list[dict]:
    esm150 = pd.read_csv(spec.esm2_150m_features, usecols=[HASH_COL, LABEL_COL])
    esm650 = pd.read_csv(spec.esm2_650m_features, usecols=[HASH_COL, LABEL_COL])
    target_order = esm150[HASH_COL].astype(str).tolist()
    clean, diagnostics = load_source(spec, target_order)
    aac, dpc = make_feature_tables(clean)

    aac_path = OUT / f"{spec.output_prefix}_aac_features.csv"
    dpc_path = OUT / f"{spec.output_prefix}_dpc_features.csv"
    aac.to_csv(aac_path, index=False)
    dpc.to_csv(dpc_path, index=False)

    rows: list[dict] = []
    for feature_name, feature_df, path in [("AAC", aac, aac_path), ("DPC", dpc, dpc_path)]:
        output_info = {
            **diagnostics,
            "Output_File": str(path),
        }
        rows.append(compare_hash_label(spec, feature_name, feature_df, "ESM2_150M_feature_matrix", esm150, LABEL_COL, output_info))
        rows.append(compare_hash_label(spec, feature_name, feature_df, "ESM2_650M_feature_matrix", esm650, LABEL_COL, output_info))
        preds = pd.read_csv(spec.frozen_predictions)
        if "Embedding_Model" not in preds.columns:
            preds["Embedding_Model"] = "ESM2_150M"
        for model in ["ESM2_150M", "ESM2_650M"]:
            model_preds = preds[preds["Embedding_Model"] == model].reset_index(drop=True)
            rows.append(compare_hash_label(spec, feature_name, feature_df, f"frozen_predictions_{model}", model_preds, "y_true", output_info))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for spec in EXTERNAL_SETS:
        all_rows.extend(process_one(spec))
    write_csv(ALIGNMENT_CSV, all_rows)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task": "Recompute external AAC/DPC matrices and verify alignment before Phase 4 controlled ablation.",
        "outputs": {
            "feature_matrix_directory": str(OUT),
            "alignment_checks": str(ALIGNMENT_CSV),
        },
        "standard_amino_acids": "".join(STANDARD_AA),
        "aac_dim": len(STANDARD_AA),
        "dpc_dim": len(DIPEPTIDES),
        "sequence_cleaning": "Uppercase and retain standard amino acids only; alignment requires same hash set as ESM2 features.",
        "alignment_rule": "AAC/DPC outputs are reordered to ESM2-150M feature matrix Sequence_Hash order; labels checked against ESM2-150M, ESM2-650M, and frozen prediction files for both models.",
        "all_checks_passed": all(bool(r["Mismatch_OK"]) for r in all_rows),
    }
    (MANIFEST_JSON).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
