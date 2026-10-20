from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
RELEASE_RESULTS = ROOT / "results"
OUT = ROOT / "results" / "analysis" / "phase1"

RANDOM_SEED = 20260713
BOOTSTRAP_B = 10000
THRESHOLD = 0.5


@dataclass(frozen=True)
class ExternalSet:
    set_id: str
    display_name: str
    file_name: str
    n_expected: int
    notes: str


EXTERNAL_SETS = [
    ExternalSet(
        set_id="strict_novel40",
        display_name="Strict Seq2Topt novel40 external",
        file_name="29_model_scale_external_results.csv",
        n_expected=66,
        notes="Most stringent low-homology external panel; CI is expected to be wide because n=66 with 26 positives and 40 negatives.",
    ),
    ExternalSet(
        set_id="seq2topt_noexact",
        display_name="Seq2Topt no-exact-overlap external",
        file_name="32_external_noexact_model_scale_results.csv",
        n_expected=136,
        notes="Larger Seq2Topt-derived panel after exact-overlap removal; not novel40-filtered.",
    ),
    ExternalSet(
        set_id="public_topt_noexact",
        display_name="Public Topt no-exact-overlap external",
        file_name="34_public_topt_noexact_model_scale_results.csv",
        n_expected=984,
        notes="Expanded public Topt panel curated from biomap-research/optimal_temperature after exact-overlap removal.",
    ),
]


METRICS = [
    "ROC_AUC",
    "PR_AUC",
    "MCC",
    "F1",
    "Balanced_Accuracy",
    "Precision",
    "Recall",
    "Brier_Score",
]


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def roc_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rank_average(y_prob)
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_prob, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true)
    fp = np.cumsum(1 - sorted_true)
    precision = tp / np.maximum(tp + fp, 1)
    return float((precision * sorted_true).sum() / n_pos)


def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD) -> tuple[int, int, int, int]:
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tn, fp, fn, tp


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_counts(y_true, y_prob)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn - fp * fn) / np.sqrt(denom)) if denom else 0.0
    return {
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(pr_auc_average_precision(y_true, y_prob)),
        "MCC": float(mcc),
        "F1": float(f1),
        "Balanced_Accuracy": float((recall + specificity) / 2.0),
        "Precision": float(precision),
        "Recall": float(recall),
        "Brier_Score": float(np.mean((y_prob - y_true) ** 2)),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
        "TP": float(tp),
    }


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    return np.concatenate([boot_pos, boot_neg])


def ci(values: np.ndarray) -> tuple[float, float, float]:
    clean = values[~np.isnan(values)]
    return (
        float(np.mean(clean)),
        float(np.percentile(clean, 2.5)),
        float(np.percentile(clean, 97.5)),
    )


def load_external_set(spec: ExternalSet) -> pd.DataFrame:
    path = RELEASE_RESULTS / spec.file_name
    df = pd.read_csv(path)
    if "Embedding_Model" not in df.columns:
        df["Embedding_Model"] = "ESM2_150M"
    df["External_Set"] = spec.set_id
    df["External_Set_Name"] = spec.display_name
    df["External_Set_Notes"] = spec.notes
    for col in ["y_true", "y_prob"]:
        if col not in df.columns:
            raise ValueError(f"{spec.file_name} missing required column: {col}")
    df["y_true"] = df["y_true"].astype(int)
    df["y_prob"] = df["y_prob"].astype(float)
    if len(df) % spec.n_expected != 0:
        raise ValueError(f"{spec.file_name} row count {len(df)} is not compatible with expected n={spec.n_expected}")
    return df


def summarize_one(set_spec: ExternalSet, model: str, df: pd.DataFrame, rng: np.random.Generator) -> tuple[list[dict], dict, dict]:
    y_true = df["y_true"].to_numpy(dtype=int)
    y_prob = df["y_prob"].to_numpy(dtype=float)
    observed = classification_metrics(y_true, y_prob)
    n = len(y_true)
    pos = int(y_true.sum())
    neg = int(n - pos)

    boot_values = {m: np.empty(BOOTSTRAP_B, dtype=float) for m in METRICS}
    for b in range(BOOTSTRAP_B):
        idx = stratified_bootstrap_indices(y_true, rng)
        vals = classification_metrics(y_true[idx], y_prob[idx])
        for metric in METRICS:
            boot_values[metric][b] = vals[metric]

    metric_rows: list[dict] = []
    for metric in METRICS:
        boot_mean, lo, hi = ci(boot_values[metric])
        metric_rows.append(
            {
                "External_Set": set_spec.set_id,
                "External_Set_Name": set_spec.display_name,
                "Embedding_Model": model,
                "N": n,
                "Positives": pos,
                "Negatives": neg,
                "Positive_Rate": pos / n,
                "Metric": metric,
                "Observed": observed[metric],
                "Bootstrap_Mean": boot_mean,
                "CI95_Lower": lo,
                "CI95_Upper": hi,
                "CI95_Width": hi - lo,
                "Bootstrap_B": BOOTSTRAP_B,
                "Bootstrap_Type": "stratified_with_replacement",
                "Decision_Threshold": THRESHOLD if metric not in {"ROC_AUC", "PR_AUC", "Brier_Score"} else "",
                "Notes": set_spec.notes,
            }
        )

    main_row = {
        "External_Set": set_spec.set_id,
        "External_Set_Name": set_spec.display_name,
        "Embedding_Model": model,
        "N": n,
        "Positives": pos,
        "Negatives": neg,
        "Positive_Rate": pos / n,
        "TN": int(observed["TN"]),
        "FP": int(observed["FP"]),
        "FN": int(observed["FN"]),
        "TP": int(observed["TP"]),
        "Bootstrap_B": BOOTSTRAP_B,
        "Bootstrap_Type": "stratified_with_replacement",
        "Thresholded_Metrics_Decision_Threshold": THRESHOLD,
        "Threshold_Selection": "fixed_a_priori_not_tuned_on_external_set",
        "Notes": set_spec.notes,
    }
    for row in metric_rows:
        metric = row["Metric"]
        main_row[f"{metric}"] = row["Observed"]
        main_row[f"{metric}_CI95"] = f"{row['CI95_Lower']:.4f}-{row['CI95_Upper']:.4f}"
        main_row[f"{metric}_CI95_Width"] = row["CI95_Width"]

    raw_boot = {
        "External_Set": set_spec.set_id,
        "Embedding_Model": model,
        "N": n,
        "Positives": pos,
        "Negatives": neg,
        "metrics": {metric: boot_values[metric].round(10).tolist() for metric in METRICS},
    }
    return metric_rows, main_row, raw_boot


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


def format_metric(row: dict, metric: str) -> str:
    return f"{row[metric]:.4f} ({row[f'{metric}_CI95']})"


def write_manuscript_text(main_rows: list[dict]) -> None:
    by_set = {(r["External_Set"], r["Embedding_Model"]): r for r in main_rows}
    lines = [
        "# Phase 1 Manuscript-Ready External Bootstrap Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Stratified bootstrap confidence intervals were computed separately within the positive and negative classes with replacement, preserving each external panel's original class counts. We used B = 10,000 bootstrap replicates and report percentile-based 95% CIs. The strict novel40 panel (N = 66; 26 positives and 40 negatives) is intentionally reported with its naturally wide intervals; no smoothing, shrinkage, or post hoc interval narrowing was applied. Threshold-dependent metrics (MCC, F1, balanced accuracy, precision, and recall) used a fixed a priori decision threshold of 0.5 and were not optimized on any external validation set.",
        "",
        "## Main External Validation Description",
        "",
    ]
    for set_id in ["strict_novel40", "seq2topt_noexact", "public_topt_noexact"]:
        rows = [r for r in main_rows if r["External_Set"] == set_id and r["Embedding_Model"] == "ESM2_150M"]
        if not rows:
            continue
        r = rows[0]
        lines.append(
            f"- {r['External_Set_Name']} ({r['N']} sequences; {r['Positives']} positives, {r['Negatives']} negatives): "
            f"ROC-AUC {format_metric(r, 'ROC_AUC')}, PR-AUC {format_metric(r, 'PR_AUC')}, "
            f"MCC {format_metric(r, 'MCC')}, F1 {format_metric(r, 'F1')}, "
            f"balanced accuracy {format_metric(r, 'Balanced_Accuracy')}, "
            f"Brier score {format_metric(r, 'Brier_Score')}."
        )
    lines += [
        "",
        "## Interpretation Note",
        "",
        "The wide confidence intervals on the strict N = 66 novel40 set should be interpreted as an honest reflection of limited statistical power after stringent homology filtering, not as a defect to be hidden. In the TCBB manuscript, this panel should remain the strictest low-homology validation evidence, while the N = 136 and N = 984 panels provide complementary larger-scale no-exact-overlap external evidence.",
        "",
        "## Recommended Table Caption",
        "",
        "External validation performance with stratified bootstrap 95% confidence intervals across strict and expanded external panels. CIs were computed using 10,000 within-class bootstrap resamples and preserve the original positive/negative counts of each external panel. Threshold-dependent metrics used a fixed decision threshold of 0.5; no external set was used for threshold optimization. The strict novel40 set is intentionally retained as the most stringent but statistically limited validation panel.",
        "",
    ]
    (OUT / "external_bootstrap_manuscript_text.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    all_metric_rows: list[dict] = []
    all_main_rows: list[dict] = []
    raw_bootstraps: list[dict] = []

    for spec in EXTERNAL_SETS:
        df = load_external_set(spec)
        for model, model_df in df.groupby("Embedding_Model", sort=True):
            metric_rows, main_row, raw_boot = summarize_one(spec, str(model), model_df.reset_index(drop=True), rng)
            all_metric_rows.extend(metric_rows)
            all_main_rows.append(main_row)
            raw_bootstraps.append(raw_boot)

    write_csv(OUT / "external_bootstrap_ci_all_sets.csv", all_metric_rows)
    write_csv(OUT / "external_validation_main_table.csv", all_main_rows)
    (OUT / "external_bootstrap_raw_values.json").write_text(json.dumps(raw_bootstraps, ensure_ascii=False), encoding="utf-8")
    write_manuscript_text(all_main_rows)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "random_seed": RANDOM_SEED,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_type": "stratified_with_replacement",
        "decision_threshold": THRESHOLD,
        "inputs": [str(RELEASE_RESULTS / spec.file_name) for spec in EXTERNAL_SETS],
        "outputs": {
            "external_bootstrap_ci_all_sets": str(OUT / "external_bootstrap_ci_all_sets.csv"),
            "external_validation_main_table": str(OUT / "external_validation_main_table.csv"),
            "external_bootstrap_raw_values": str(OUT / "external_bootstrap_raw_values.json"),
            "manuscript_text": str(OUT / "external_bootstrap_manuscript_text.md"),
        },
        "metrics": METRICS,
        "important_note": "Strict N=66 CIs are expected to be wide and are not narrowed or smoothed.",
    }
    (OUT / "phase1_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
