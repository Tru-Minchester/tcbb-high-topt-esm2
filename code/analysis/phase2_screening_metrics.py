from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
RELEASE = ROOT
PUBLIC_DATA = RELEASE / "data_processed" / "33_public_topt_candidate_noexact.csv"
PUBLIC_PREDICTIONS = RELEASE / "results" / "34_public_topt_noexact_model_scale_results.csv"
OUT = ROOT / "results" / "analysis" / "phase2"

RANDOM_SEED = 20260713
N_REPLICATES = 1000
TOP_FRACTIONS = [0.01, 0.05, 0.10]
RATIOS = [20, 50, 100]
TOPK_ROUNDING = "ceil_min1"
RELIABILITY_MIN_TOPK_SLOTS = 5
UNDERPOWERED_WIDTH_FRACTION = 0.5


@dataclass(frozen=True)
class DecoyCondition:
    condition: str
    description: str
    min_temp: float | None
    max_temp: float
    manuscript_role: str


DECOY_CONDITIONS = [
    DecoyCondition(
        condition="broad_lt60",
        description="Broad external decoys with Topt < 60 degrees C",
        min_temp=None,
        max_temp=59.999,
        manuscript_role="main",
    ),
    DecoyCondition(
        condition="hard_40_59",
        description="Near-boundary hard external decoys with 40 <= Topt < 60 degrees C",
        min_temp=40.0,
        max_temp=59.999,
        manuscript_role="main",
    ),
    DecoyCondition(
        condition="hard_50_59",
        description="Very hard near-threshold external decoys with 50 <= Topt < 60 degrees C",
        min_temp=50.0,
        max_temp=59.999,
        manuscript_role="supplementary_stress_test",
    ),
]


def top_k_count(pool_size: int, frac: float) -> int:
    return max(1, math.ceil(pool_size * frac))


def reliability_label(pool_size: int, frac: float) -> str:
    slots = top_k_count(pool_size, frac)
    return "main" if slots >= RELIABILITY_MIN_TOPK_SLOTS else "exploratory"


def percentile_ci(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.mean(values)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def load_inputs() -> pd.DataFrame:
    data = pd.read_csv(PUBLIC_DATA)
    preds = pd.read_csv(PUBLIC_PREDICTIONS)
    data["Binary_Label"] = data["Binary_Label"].astype(int)
    data["Temp_Mean"] = data["Temp_Mean"].astype(float)
    preds["y_true"] = preds["y_true"].astype(int)
    preds["y_prob"] = preds["y_prob"].astype(float)
    merged = preds.merge(
        data[
            [
                "Sequence_Hash",
                "Binary_Label",
                "Temp_Mean",
                "Temp_Min",
                "Temp_Max",
                "Sequence_Length",
                "Source_DB",
                "Source_Splits",
            ]
        ],
        on="Sequence_Hash",
        how="left",
        validate="many_to_one",
    )
    if merged["Binary_Label"].isna().any():
        missing = int(merged["Binary_Label"].isna().sum())
        raise ValueError(f"Missing public data metadata for {missing} prediction rows")
    mismatch = (merged["Binary_Label"].astype(int) != merged["y_true"].astype(int)).sum()
    if mismatch:
        raise ValueError(f"Prediction y_true / public Binary_Label mismatches: {mismatch}")
    return merged


def pool_size_for_ratio(n_pos_available: int, n_decoy_available: int, ratio: int) -> tuple[int, int]:
    n_pos = min(n_pos_available, n_decoy_available // ratio)
    n_decoy = n_pos * ratio
    if n_pos < 1 or n_decoy < 1:
        raise ValueError(f"Insufficient positives/decoys for ratio 1:{ratio}")
    return n_pos, n_decoy


def summarize_pool_design(df: pd.DataFrame) -> list[dict]:
    base = df[df["Embedding_Model"] == "ESM2_150M"].copy()
    positives = base[base["y_true"] == 1]
    negatives = base[base["y_true"] == 0]
    rows = []
    for cond in DECOY_CONDITIONS:
        decoys = negatives[negatives["Temp_Mean"] <= cond.max_temp]
        if cond.min_temp is not None:
            decoys = decoys[decoys["Temp_Mean"] >= cond.min_temp]
        for ratio in RATIOS:
            n_pos, n_decoy = pool_size_for_ratio(len(positives), len(decoys), ratio)
            pool_size = n_pos + n_decoy
            base_rate = n_pos / pool_size
            ef_max = 1.0 / base_rate
            for frac in TOP_FRACTIONS:
                topk_slots = top_k_count(pool_size, frac)
                rows.append(
                    {
                        "Decoy_Condition": cond.condition,
                        "Decoy_Description": cond.description,
                        "Manuscript_Role": cond.manuscript_role,
                        "Ratio_Positive_to_Negative": f"1:{ratio}",
                        "Ratio_Negatives_Per_Positive": ratio,
                        "Available_Positives": len(positives),
                        "Available_Decoys": len(decoys),
                        "Positives_Per_Pool": n_pos,
                        "Decoys_Per_Pool": n_decoy,
                        "Pool_Size": pool_size,
                        "Base_Positive_Rate": base_rate,
                        "EF_Max": ef_max,
                        "Top_Fraction": frac,
                        "TopK_Slots": topk_slots,
                        "TopK_Rounding": TOPK_ROUNDING,
                        "Reliability": reliability_label(pool_size, frac),
                    }
                )
    return rows


def compute_replicate_metrics(
    model_df: pd.DataFrame,
    cond: DecoyCondition,
    ratio: int,
    replicate: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], dict]:
    positives = model_df[model_df["y_true"] == 1]
    negatives = model_df[model_df["y_true"] == 0]
    decoys = negatives[negatives["Temp_Mean"] <= cond.max_temp]
    if cond.min_temp is not None:
        decoys = decoys[decoys["Temp_Mean"] >= cond.min_temp]

    n_pos, n_decoy = pool_size_for_ratio(len(positives), len(decoys), ratio)
    sampled_pos_idx = rng.choice(positives.index.to_numpy(), size=n_pos, replace=False)
    sampled_decoy_idx = rng.choice(decoys.index.to_numpy(), size=n_decoy, replace=False)
    pool = model_df.loc[np.concatenate([sampled_pos_idx, sampled_decoy_idx])].copy()
    pool = pool.sort_values("y_prob", ascending=False, kind="mergesort").reset_index(drop=True)

    pool_size = len(pool)
    base_rate = n_pos / pool_size
    ef_max = 1.0 / base_rate
    model = str(model_df["Embedding_Model"].iloc[0])
    rows: list[dict] = []

    for frac in TOP_FRACTIONS:
        k = top_k_count(pool_size, frac)
        top = pool.iloc[:k]
        hits = int(top["y_true"].sum())
        precision = hits / k
        recall = hits / n_pos
        ef_observed = precision / base_rate if base_rate else float("nan")
        ef_normalized = ef_observed / ef_max if ef_max else float("nan")
        rows.append(
            {
                "Embedding_Model": model,
                "Analysis_Role": "primary" if model == "ESM2_150M" else "model_scale_sensitivity",
                "Decoy_Condition": cond.condition,
                "Decoy_Description": cond.description,
                "Manuscript_Role": cond.manuscript_role,
                "Ratio_Positive_to_Negative": f"1:{ratio}",
                "Ratio_Negatives_Per_Positive": ratio,
                "Replicate": replicate,
                "Pool_Size": pool_size,
                "Positives_In_Pool": n_pos,
                "Decoys_In_Pool": n_decoy,
                "Base_Positive_Rate": base_rate,
                "EF_Max": ef_max,
                "Top_Fraction": frac,
                "TopK_Slots": k,
                "TopK_Rounding": TOPK_ROUNDING,
                "Reliability": reliability_label(pool_size, frac),
                "Hits_In_TopK": hits,
                "Precision_TopK": precision,
                "Recall_TopK": recall,
                "EF_Observed": ef_observed,
                "EF_Normalized": ef_normalized,
            }
        )

    curve_rows: list[dict] = []
    cumulative_hits = np.cumsum(pool["y_true"].to_numpy(dtype=int))
    for pct in range(1, 101):
        frac = pct / 100
        k = top_k_count(pool_size, frac)
        hits = int(cumulative_hits[k - 1])
        curve_rows.append(
            {
                "Embedding_Model": model,
                "Analysis_Role": "primary" if model == "ESM2_150M" else "model_scale_sensitivity",
                "Decoy_Condition": cond.condition,
                "Ratio_Positive_to_Negative": f"1:{ratio}",
                "Ratio_Negatives_Per_Positive": ratio,
                "Replicate": replicate,
                "Pool_Size": pool_size,
                "Positives_In_Pool": n_pos,
                "Decoys_In_Pool": n_decoy,
                "Base_Positive_Rate": base_rate,
                "Top_Fraction": frac,
                "TopK_Slots": k,
                "TopK_Rounding": TOPK_ROUNDING,
                "Hits_Recovered": hits,
                "Recall": hits / n_pos,
                "Precision": hits / k,
            }
        )

    manifest_row = {
        "Embedding_Model": model,
        "Analysis_Role": "primary" if model == "ESM2_150M" else "model_scale_sensitivity",
        "Decoy_Condition": cond.condition,
        "Decoy_Description": cond.description,
        "Manuscript_Role": cond.manuscript_role,
        "Ratio_Positive_to_Negative": f"1:{ratio}",
        "Ratio_Negatives_Per_Positive": ratio,
        "Replicate": replicate,
        "Pool_Size": pool_size,
        "Positives_In_Pool": n_pos,
        "Decoys_In_Pool": n_decoy,
        "Base_Positive_Rate": base_rate,
        "EF_Max": ef_max,
        "TopK_Rounding": TOPK_ROUNDING,
        "Positive_Sampling": "without_replacement_within_replicate",
        "Decoy_Sampling": "without_replacement_within_replicate",
        "Reuse_Across_Replicates": "allowed",
        "Random_Seed": RANDOM_SEED,
    }
    return rows, curve_rows, manifest_row


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


def compute_ci_table(metrics: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    group_cols = [
        "Embedding_Model",
        "Analysis_Role",
        "Decoy_Condition",
        "Manuscript_Role",
        "Ratio_Positive_to_Negative",
        "Ratio_Negatives_Per_Positive",
        "Top_Fraction",
        "TopK_Slots",
        "Reliability",
        "Pool_Size",
        "Positives_In_Pool",
        "Decoys_In_Pool",
        "Base_Positive_Rate",
        "EF_Max",
    ]
    value_cols = ["Precision_TopK", "Recall_TopK", "EF_Observed", "EF_Normalized", "Hits_In_TopK"]
    for keys, group in metrics.groupby(group_cols, sort=True, dropna=False):
        base = dict(zip(group_cols, keys))
        for metric in value_cols:
            values = group[metric].to_numpy(dtype=float)
            mean, lo, hi = percentile_ci(values)
            width = hi - lo
            underpowered = False
            if metric == "EF_Observed":
                underpowered = (width / float(base["EF_Max"])) > UNDERPOWERED_WIDTH_FRACTION
            rows.append(
                {
                    **base,
                    "Metric": metric,
                    "Mean": mean,
                    "CI95_Lower": lo,
                    "CI95_Upper": hi,
                    "CI95_Width": width,
                    "Bootstrap_B": N_REPLICATES,
                    "CI_Method": "percentile_over_repeated_spiked_pool_replicates",
                    "Underpowered": bool(underpowered),
                    "Underpowered_Rule": f"EF CI width / EF_Max > {UNDERPOWERED_WIDTH_FRACTION}" if metric == "EF_Observed" else "",
                    "TopK_Rounding": TOPK_ROUNDING,
                }
            )
    return rows


def compute_main_table(ci_rows: list[dict]) -> list[dict]:
    ci = pd.DataFrame(ci_rows)
    index_cols = [
        "Embedding_Model",
        "Analysis_Role",
        "Decoy_Condition",
        "Manuscript_Role",
        "Ratio_Positive_to_Negative",
        "Ratio_Negatives_Per_Positive",
        "Top_Fraction",
        "TopK_Slots",
        "Reliability",
        "Pool_Size",
        "Positives_In_Pool",
        "Decoys_In_Pool",
        "Base_Positive_Rate",
        "EF_Max",
    ]
    rows: list[dict] = []
    for keys, group in ci.groupby(index_cols, sort=True, dropna=False):
        base = dict(zip(index_cols, keys))
        metric_rows = {r["Metric"]: r for _, r in group.iterrows()}
        ef_row = metric_rows.get("EF_Observed", {})
        precision_row = metric_rows.get("Precision_TopK", {})
        recall_row = metric_rows.get("Recall_TopK", {})
        hits_row = metric_rows.get("Hits_In_TopK", {})
        top_fraction = float(base["Top_Fraction"])
        ef_underpowered = str(ef_row.get("Underpowered", "")) == "False"
        rows.append(
            {
                **base,
                "EF_Mean": ef_row.get("Mean", ""),
                "EF_CI95_Lower": ef_row.get("CI95_Lower", ""),
                "EF_CI95_Upper": ef_row.get("CI95_Upper", ""),
                "EF_CI95_Width": ef_row.get("CI95_Width", ""),
                "EF_Underpowered": ef_row.get("Underpowered", ""),
                "Precision_TopK_Mean": precision_row.get("Mean", ""),
                "Precision_TopK_CI95_Lower": precision_row.get("CI95_Lower", ""),
                "Precision_TopK_CI95_Upper": precision_row.get("CI95_Upper", ""),
                "Recall_TopK_Mean": recall_row.get("Mean", ""),
                "Recall_TopK_CI95_Lower": recall_row.get("CI95_Lower", ""),
                "Recall_TopK_CI95_Upper": recall_row.get("CI95_Upper", ""),
                "Hits_In_TopK_Mean": hits_row.get("Mean", ""),
                "TopK_Rounding": TOPK_ROUNDING,
                "Main_Text_Eligible": (
                    base["Embedding_Model"] == "ESM2_150M"
                    and base["Reliability"] == "main"
                    and base["Decoy_Condition"] == "broad_lt60"
                    and top_fraction in {0.05, 0.10}
                    and ef_underpowered
                ),
                "Presentation_Tier": (
                    "main_text"
                    if (
                        base["Embedding_Model"] == "ESM2_150M"
                        and base["Reliability"] == "main"
                        and base["Decoy_Condition"] == "broad_lt60"
                        and top_fraction in {0.05, 0.10}
                        and ef_underpowered
                    )
                    else "supplementary_sensitivity"
                ),
                "Presentation_Reason": (
                    "Primary broad-decoy Top-5/10% screening result"
                    if (
                        base["Embedding_Model"] == "ESM2_150M"
                        and base["Reliability"] == "main"
                        and base["Decoy_Condition"] == "broad_lt60"
                        and top_fraction in {0.05, 0.10}
                        and ef_underpowered
                    )
                    else "Excluded from main claims because Top-1% resolution is limited, condition is a hard-decoy sensitivity analysis, model is 650M sensitivity, reliability is exploratory, or EF CI is underpowered"
                ),
            }
        )
    return rows


def compute_curve_summary(curve_rows: list[dict]) -> list[dict]:
    curve = pd.DataFrame(curve_rows)
    group_cols = [
        "Embedding_Model",
        "Analysis_Role",
        "Decoy_Condition",
        "Ratio_Positive_to_Negative",
        "Ratio_Negatives_Per_Positive",
        "Top_Fraction",
        "Pool_Size",
        "Positives_In_Pool",
        "Decoys_In_Pool",
        "Base_Positive_Rate",
        "TopK_Slots",
    ]
    rows: list[dict] = []
    for keys, group in curve.groupby(group_cols, sort=True, dropna=False):
        base = dict(zip(group_cols, keys))
        recall_mean, recall_lo, recall_hi = percentile_ci(group["Recall"].to_numpy(dtype=float))
        precision_mean, precision_lo, precision_hi = percentile_ci(group["Precision"].to_numpy(dtype=float))
        hits_mean, hits_lo, hits_hi = percentile_ci(group["Hits_Recovered"].to_numpy(dtype=float))
        rows.append(
            {
                **base,
                "Recall_Mean": recall_mean,
                "Recall_CI95_Lower": recall_lo,
                "Recall_CI95_Upper": recall_hi,
                "Precision_Mean": precision_mean,
                "Precision_CI95_Lower": precision_lo,
                "Precision_CI95_Upper": precision_hi,
                "Hits_Recovered_Mean": hits_mean,
                "Hits_Recovered_CI95_Lower": hits_lo,
                "Hits_Recovered_CI95_Upper": hits_hi,
                "TopK_Rounding": TOPK_ROUNDING,
            }
        )
    return rows


def write_manuscript_text(ci_rows: list[dict]) -> None:
    main_table = pd.DataFrame(compute_main_table(ci_rows))
    primary = main_table[main_table["Main_Text_Eligible"] == True].copy()
    underpowered = main_table[
        (main_table["Embedding_Model"] == "ESM2_150M")
        & (main_table["Manuscript_Role"] == "main")
        & (main_table["EF_Underpowered"] == True)
    ].copy()
    lines = [
        "# Phase 2 Manuscript-Ready Screening Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "The primary screening analysis uses ESM2-150M to preserve comparability with the PLOS baseline (Figure 5 / Table 3). ESM2-650M is reported as a model-scale sensitivity analysis; screening trends should be interpreted across scales rather than used to redefine the baseline model.",
        "",
        "Repeated spiked-in screening pools were constructed from the public no-exact-overlap Topt external panel. Positives were sampled from external high-Topt sequences (Topt >= 60 degrees C), and decoys were sampled from external low-Topt sequences. Broad decoys (Topt < 60 degrees C) and hard near-threshold decoys (40 <= Topt < 60 degrees C) are treated as manuscript-level analyses; 50-59 degrees C decoys are retained as a supplementary stress test because available pool sizes are smaller.",
        "",
        "Top-K cutoffs used a single pre-specified rule, `ceil_min1`: top_k_count = max(1, ceil(pool_size * fraction)). For each pool, EF_max = 1 / base_positive_rate. EF_observed / EF_max can be interpreted as the fraction of the theoretical enrichment limit reached, but because this quantity is algebraically equivalent to Precision@TopK, it is not reported as an independent headline metric.",
        "",
        "At the current external pool sizes, Top-1% corresponds to only 4-7 ranked positions in most manuscript-level pools and 2 positions in the hardest 50-59 degrees C stress-test pools. These Top-1% results are therefore treated as resolution-limited and reported only for completeness. Main screening conclusions are based on Top-5% and Top-10% broad-decoy pools. The hard 40-59 and hard 50-59 decoy settings are retained as supplementary sensitivity analyses because small positive counts and near-threshold decoys often yield wide intervals or lower confidence-boundary values at zero.",
        "",
        "## Main EF Results To Consider For Manuscript",
        "",
    ]
    if primary.empty:
        lines.append("No primary/main EF rows were generated.")
    else:
        for _, row in primary.sort_values(["Decoy_Condition", "Ratio_Negatives_Per_Positive", "Top_Fraction"]).iterrows():
            lines.append(
                f"- {row['Decoy_Condition']}, ratio {row['Ratio_Positive_to_Negative']}, Top-{int(float(row['Top_Fraction']) * 100)}%: "
                f"EF {float(row['EF_Mean']):.2f} (95% CI {float(row['EF_CI95_Lower']):.2f}-{float(row['EF_CI95_Upper']):.2f}); "
                f"EF_max {float(row['EF_Max']):.2f}; "
                f"precision {float(row['Precision_TopK_Mean']):.2f}; recall {float(row['Recall_TopK_Mean']):.2f}."
            )
    lines += [
        "",
        "Rows marked `exploratory` or `underpowered=True` are excluded from the main claims and should be described as supplementary sensitivity evidence or reported for completeness.",
        "",
    ]
    if not underpowered.empty:
        lines += ["## Underpowered Conditions Reported For Completeness", ""]
        for _, row in underpowered.sort_values(["Decoy_Condition", "Ratio_Negatives_Per_Positive", "Top_Fraction"]).iterrows():
            lines.append(
                f"- {row['Decoy_Condition']}, ratio {row['Ratio_Positive_to_Negative']}, Top-{int(float(row['Top_Fraction']) * 100)}%: "
                f"EF CI width {float(row['EF_CI95_Width']):.2f} exceeds 50% of EF_max {float(row['EF_Max']):.2f}."
            )
        lines.append("")
    (OUT / "screening_manuscript_text.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    df = load_inputs()

    design_rows = summarize_pool_design(df)
    write_csv(OUT / "screening_pool_design.csv", design_rows)

    metric_rows: list[dict] = []
    curve_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for model, model_df in df.groupby("Embedding_Model", sort=True):
        model_df = model_df.reset_index(drop=True)
        for cond in DECOY_CONDITIONS:
            for ratio in RATIOS:
                for replicate in range(1, N_REPLICATES + 1):
                    rows, curve, manifest = compute_replicate_metrics(model_df, cond, ratio, replicate, rng)
                    metric_rows.extend(rows)
                    curve_rows.extend(curve)
                    manifest_rows.append(manifest)

    write_csv(OUT / "screening_metrics_spiked_pools.csv", metric_rows)
    write_csv(OUT / "hit_recovery_curve.csv", curve_rows)
    write_csv(OUT / "screening_pool_manifest.csv", manifest_rows)

    ci_rows = compute_ci_table(pd.DataFrame(metric_rows))
    write_csv(OUT / "screening_bootstrap_ci.csv", ci_rows)
    main_table_rows = compute_main_table(ci_rows)
    write_csv(OUT / "screening_main_table.csv", main_table_rows)
    curve_summary_rows = compute_curve_summary(curve_rows)
    write_csv(OUT / "hit_recovery_curve_summary.csv", curve_summary_rows)
    write_manuscript_text(ci_rows)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "random_seed": RANDOM_SEED,
        "n_replicates": N_REPLICATES,
        "top_fractions": TOP_FRACTIONS,
        "ratios_positive_to_negative": [f"1:{r}" for r in RATIOS],
        "topk_rounding": TOPK_ROUNDING,
        "topk_function": "top_k_count(pool_size, frac) = max(1, ceil(pool_size * frac))",
        "reliability_rule": f"main if top_k_slots >= {RELIABILITY_MIN_TOPK_SLOTS}, else exploratory",
        "underpowered_rule": f"EF_Observed CI95 width / EF_Max > {UNDERPOWERED_WIDTH_FRACTION}",
        "primary_model": "ESM2_150M",
        "sensitivity_model": "ESM2_650M",
        "inputs": {
            "public_data": str(PUBLIC_DATA),
            "public_predictions": str(PUBLIC_PREDICTIONS),
        },
        "outputs": {
            "screening_pool_design": str(OUT / "screening_pool_design.csv"),
            "screening_metrics_spiked_pools": str(OUT / "screening_metrics_spiked_pools.csv"),
            "screening_bootstrap_ci": str(OUT / "screening_bootstrap_ci.csv"),
            "screening_main_table": str(OUT / "screening_main_table.csv"),
            "hit_recovery_curve": str(OUT / "hit_recovery_curve.csv"),
            "hit_recovery_curve_summary": str(OUT / "hit_recovery_curve_summary.csv"),
            "screening_pool_manifest": str(OUT / "screening_pool_manifest.csv"),
            "screening_manuscript_text": str(OUT / "screening_manuscript_text.md"),
        },
    }
    (OUT / "phase2_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
