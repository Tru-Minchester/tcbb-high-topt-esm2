from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
RELEASE = ROOT
INTERNAL_EC = RELEASE / "data_processed" / "13_cdhit40_strict_nomixed_dataset.csv"
PHASE4_PRED = ROOT / "results" / "analysis" / "phase4" / "predictions" / "phase4_predictions_all_branches.csv"
PHASE4_INTERNAL = ROOT / "results" / "analysis" / "phase4" / "fusion_ablation_internal.csv"
PHASE6_INVENTORY = ROOT / "results" / "analysis" / "phase6" / "phase6_ec_inventory.csv"

OUT_CSV = ROOT / "results" / "analysis" / "phase6" / "phase6_ec_stratified_metrics.csv"
OUT_TEXT = ROOT / "results" / "analysis" / "phase6" / "phase6_ec_stratified_text.md"
OUT_MANIFEST = ROOT / "results" / "analysis" / "phase6" / "phase6_ec_stratified_manifest.json"

HASH_COL = "Sequence_Hash"
LABEL_COL = "Binary_Label"
EC_COL = "Representative_EC_Number"
THRESHOLD = 0.5
LOW_POWER_POS_CUTOFF = 15
LOW_POWER_N_CUTOFF = 100

EC_LEVEL1_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}

MODEL_BRANCHES = [
    ("ESM2-150M", "primary"),
    ("ESM2-650M", "model_scale_sensitivity"),
]


def ec_level1(value: Any) -> str:
    if pd.isna(value):
        return ""
    match = re.search(r"\b([1-7])\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|-)\b", str(value))
    return match.group(1) if match else ""


def metric_values(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= THRESHOLD).astype(int)
    return {
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "PR_AUC": float(average_precision_score(y_true, y_prob)) if int(y_true.sum()) > 0 else np.nan,
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) == 2 else np.nan,
        "Brier": float(np.mean((y_prob.astype(float) - y_true.astype(float)) ** 2)),
    }


def power_tier(n: int, pos: int, level: str) -> tuple[str, str]:
    if pos < LOW_POWER_POS_CUTOFF or n < LOW_POWER_N_CUTOFF:
        return "low_power_exploratory", f"positives<{LOW_POWER_POS_CUTOFF} or N<{LOW_POWER_N_CUTOFF}"
    if level in {"1", "2", "3"}:
        return "main_sufficient_sample", "EC1-EC3 are treated as primary sufficiently represented classes"
    return "reported_medium_sample", "sample size adequate but smaller than EC1-EC3"


def load_joined(branch: str) -> pd.DataFrame:
    ec = pd.read_csv(INTERNAL_EC, usecols=[HASH_COL, LABEL_COL, EC_COL])
    ec[HASH_COL] = ec[HASH_COL].astype(str)
    ec[LABEL_COL] = pd.to_numeric(ec[LABEL_COL], errors="coerce").astype(int)
    ec["EC_Level1"] = ec[EC_COL].apply(ec_level1)

    pred = pd.read_csv(PHASE4_PRED)
    pred = pred[pred["Dataset"].eq("internal_oof") & pred["Branch"].eq(branch)].copy()
    pred[HASH_COL] = pred[HASH_COL].astype(str)
    pred["y_true"] = pred["y_true"].astype(int)
    pred["y_prob"] = pred["y_prob"].astype(float)

    joined = pred.merge(ec, on=HASH_COL, how="inner", validate="one_to_one")
    label_mismatch = int((joined["y_true"].astype(int) != joined[LABEL_COL].astype(int)).sum())
    if len(joined) != len(ec) or len(joined) != len(pred) or label_mismatch:
        raise ValueError(
            f"{branch} EC/prediction join failed: joined={len(joined)}, ec={len(ec)}, pred={len(pred)}, label_mismatch={label_mismatch}"
        )
    return joined


def stratified_rows() -> tuple[list[dict], dict[str, Any]]:
    rows: list[dict] = []
    validation: dict[str, Any] = {}
    phase4_internal = pd.read_csv(PHASE4_INTERNAL)

    for branch, role in MODEL_BRANCHES:
        joined = load_joined(branch)
        pooled_metrics = metric_values(joined["y_true"].to_numpy(int), joined["y_prob"].to_numpy(float))
        phase4_row = phase4_internal[phase4_internal["Branch"].eq(branch)].iloc[0]
        phase4_compare = {}
        for metric in ["ROC_AUC", "PR_AUC", "MCC", "F1", "Balanced_Accuracy", "Brier"]:
            phase4_compare[metric] = {
                "pooled_from_ec_partition": pooled_metrics[metric],
                "phase4_internal": float(phase4_row[metric]),
                "abs_diff": abs(pooled_metrics[metric] - float(phase4_row[metric])),
            }
        validation[branch] = {
            "N_sum": int(len(joined)),
            "Positives_sum": int(joined["y_true"].sum()),
            "Negatives_sum": int(len(joined) - joined["y_true"].sum()),
            "Expected_N": 5263,
            "Expected_Positives": 672,
            "Expected_Negatives": 4591,
            "Sample_Counts_Match": bool(len(joined) == 5263 and int(joined["y_true"].sum()) == 672),
            "Pooled_Metric_Comparison_To_Phase4": phase4_compare,
            "Pooled_Metrics_Match_Phase4": all(v["abs_diff"] <= 1e-12 for v in phase4_compare.values()),
            "Note": "Pooled metrics are recomputed from all EC-partitioned samples; support-weighted means of per-EC ROC-AUC/PR-AUC are not mathematically expected to equal global ROC-AUC/PR-AUC.",
        }

        for level in [str(i) for i in range(1, 8)]:
            g = joined[joined["EC_Level1"].eq(level)].copy()
            y = g["y_true"].to_numpy(int)
            p = g["y_prob"].to_numpy(float)
            n = len(g)
            pos = int(y.sum())
            neg = int(n - pos)
            tier, tier_reason = power_tier(n, pos, level)
            rows.append(
                {
                    "Dataset": "internal_strict_nomixed_oof",
                    "Model_Branch": branch,
                    "Analysis_Role": role,
                    "EC_Level1": level,
                    "EC_Level1_Name": EC_LEVEL1_NAMES[level],
                    "N": n,
                    "Positives": pos,
                    "Negatives": neg,
                    "Positive_Rate": pos / n if n else "",
                    "Power_Tier": tier,
                    "Power_Tier_Rule": tier_reason,
                    "Threshold": THRESHOLD,
                    **metric_values(y, p),
                }
            )
    return rows, validation


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


def write_text(rows: list[dict], validation: dict[str, Any]) -> None:
    df = pd.DataFrame(rows)
    primary = df[df["Model_Branch"].eq("ESM2-150M")].copy()
    ec1 = primary[primary["EC_Level1"].eq("1")].iloc[0]
    ec2 = primary[primary["EC_Level1"].eq("2")].iloc[0]
    ec3 = primary[primary["EC_Level1"].eq("3")].iloc[0]
    ec7 = primary[primary["EC_Level1"].eq("7")].iloc[0]
    external_coverage = pd.read_csv(PHASE6_INVENTORY)
    external_coverage = external_coverage[
        external_coverage["Row_Type"].eq("dataset_coverage")
        & external_coverage["Dataset_ID"].isin(["strict_novel40_N66", "seq2topt_noexact_N136", "public_topt_noexact_N984"])
    ]

    lines = [
        "# Phase 6 EC-Stratified Internal Generalization Text",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "EC-stratified analysis was performed only on the internal strict no-mixed benchmark because the three frozen external sets had 0% EC coverage in the available data files. No UniProt backfill was performed. The analysis reuses the Phase 4 frozen ESM2-150M OOF predictions as the primary model and ESM2-650M as a model-scale sensitivity analysis; no model was retrained.",
        "",
        "## Primary ESM2-150M Results",
        "",
        f"The model covers all seven EC level-1 classes. The three largest classes, EC1 oxidoreductases (N = {ec1['N']}, positives = {ec1['Positives']}), EC2 transferases (N = {ec2['N']}, positives = {ec2['Positives']}), and EC3 hydrolases (N = {ec3['N']}, positives = {ec3['Positives']}), are treated as the primary sufficiently powered strata. Their ROC-AUC values were {ec1['ROC_AUC']:.4f}, {ec2['ROC_AUC']:.4f}, and {ec3['ROC_AUC']:.4f}, respectively, with PR-AUC values {ec1['PR_AUC']:.4f}, {ec2['PR_AUC']:.4f}, and {ec3['PR_AUC']:.4f}.",
        "",
        "EC4, EC5, and EC6 were reported as medium-sample strata rather than headline claims. EC7 translocases had only N = "
        f"{ec7['N']} sequences and {ec7['Positives']} positives, and was therefore automatically marked low_power/exploratory under the rule positives < {LOW_POWER_POS_CUTOFF} or N < {LOW_POWER_N_CUTOFF}.",
        "",
        "Overall, there was no evidence of a single EC level-1 class driving the entire internal result or of a severe systematic failure concentrated in one sufficiently powered major class. EC7 should not be over-interpreted because it contains only seven positive examples.",
        "",
        "## Self-Consistency",
        "",
    ]
    for branch in ["ESM2-150M", "ESM2-650M"]:
        v = validation[branch]
        lines.append(
            f"- {branch}: EC strata sum to N = {v['N_sum']} with {v['Positives_sum']} positives and {v['Negatives_sum']} negatives; pooled recomputation matches the Phase 4 internal row within 1e-12 tolerance = {v['Pooled_Metrics_Match_Phase4']}."
        )
    lines += [
        "",
        "## External Scope Limitation",
        "",
    ]
    for _, r in external_coverage.iterrows():
        lines.append(
            f"- {r['Dataset_ID']}: EC coverage {int(r['Any_EC_Covered_N'])}/{int(r['N'])} ({float(r['Any_EC_Coverage']):.1%}); no external EC stratification was performed."
        )
    OUT_TEXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows, validation = stratified_rows()
    write_csv(OUT_CSV, rows)
    write_text(rows, validation)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis": "Phase 6 internal EC level-1 stratified metrics; no retraining.",
        "framework": {
            "predictions": str(PHASE4_PRED),
            "primary_model": "ESM2-150M internal OOF from Phase 4 controlled ablation",
            "sensitivity_model": "ESM2-650M internal OOF from Phase 4 controlled ablation",
            "threshold": THRESHOLD,
            "split": "Phase 4 seed=42 stratified 5-fold OOF",
        },
        "low_power_rule": {
            "low_power_if_positives_lt": LOW_POWER_POS_CUTOFF,
            "low_power_if_N_lt": LOW_POWER_N_CUTOFF,
            "EC7_expected_low_power": True,
        },
        "external_ec_policy": "External frozen sets have 0% EC coverage in Phase 6 inventory; no UniProt backfill and no external EC stratification.",
        "self_consistency_validation": validation,
        "outputs": {
            "phase6_ec_stratified_metrics": str(OUT_CSV),
            "text": str(OUT_TEXT),
            "manifest": str(OUT_MANIFEST),
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
