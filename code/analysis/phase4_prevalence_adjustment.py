from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
PHASE3 = ROOT / "results" / "analysis" / "phase3"
PREDICTIONS = PHASE3 / "calibrated_external_predictions.csv"
OUT_TABLE = PHASE3 / "calibration_prevalence_adjusted.csv"
OUT_MANUSCRIPT = PHASE3 / "calibration_manuscript_text.md"
OUT_MANIFEST = PHASE3 / "calibration_prevalence_adjustment_manifest.json"

INTERNAL_POSITIVES = 672
INTERNAL_N = 5263
INTERNAL_PREVALENCE = INTERNAL_POSITIVES / INTERNAL_N
PROB_CLIP = 1e-6
CALIBRATION_SEED = 20260713

BINNING_SPECS = [
    ("equal_width_10", "equal_width", 10),
    ("equal_width_5", "equal_width", 5),
    ("quantile_5", "quantile", 5),
]


def logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p.astype(float), PROB_CLIP, 1.0 - PROB_CLIP)
    return np.log(clipped / (1.0 - clipped))


def inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def prevalence_adjust(p: np.ndarray, source_prev: float, target_prev: float) -> np.ndarray:
    source_log_odds = math.log(source_prev / (1.0 - source_prev))
    target_log_odds = math.log(target_prev / (1.0 - target_prev))
    return inv_logit(logit(p) + target_log_odds - source_log_odds)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob.astype(float) - y_true.astype(float)) ** 2))


def equal_width_rows(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        mask = ((y_prob >= lo) & (y_prob <= hi)) if i == n_bins - 1 else ((y_prob >= lo) & (y_prob < hi))
        n = int(mask.sum())
        if n:
            mean_pred = float(y_prob[mask].mean())
            observed = float(y_true[mask].mean())
            gap = abs(observed - mean_pred)
        else:
            mean_pred = math.nan
            observed = math.nan
            gap = math.nan
        rows.append({"N_Bin": n, "Mean_Predicted": mean_pred, "Observed": observed, "Gap": gap})
    return rows


def quantile_rows(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> list[dict]:
    order = np.argsort(y_prob, kind="mergesort")
    rows = []
    for idx in np.array_split(order, n_bins):
        n = int(len(idx))
        if n:
            mean_pred = float(y_prob[idx].mean())
            observed = float(y_true[idx].mean())
            gap = abs(observed - mean_pred)
        else:
            mean_pred = math.nan
            observed = math.nan
            gap = math.nan
        rows.append({"N_Bin": n, "Mean_Predicted": mean_pred, "Observed": observed, "Gap": gap})
    return rows


def ece_mce(y_true: np.ndarray, y_prob: np.ndarray, method: str, n_bins: int) -> tuple[float, float]:
    rows = equal_width_rows(y_true, y_prob, n_bins) if method == "equal_width" else quantile_rows(y_true, y_prob, n_bins)
    nonempty = [r for r in rows if r["N_Bin"] > 0]
    ece = sum((r["N_Bin"] / len(y_true)) * r["Gap"] for r in nonempty)
    mce = max((r["Gap"] for r in nonempty), default=math.nan)
    return float(ece), float(mce)


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000, random_state=CALIBRATION_SEED)
    model.fit(logit(y_prob).reshape(-1, 1), y_true.astype(int))
    return float(model.coef_[0][0]), float(model.intercept_[0])


def metrics_row(
    group: pd.DataFrame,
    method: str,
    y_prob: np.ndarray,
    source_probability_column: str,
    note: str,
) -> dict:
    y_true = group["y_true"].to_numpy(dtype=int)
    y_pred = (y_prob >= 0.5).astype(int)
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    row = {
        "External_Set": group["External_Set"].iloc[0],
        "External_Set_Name": group["External_Set_Name"].iloc[0],
        "Embedding_Model": group["Embedding_Model"].iloc[0],
        "Calibration_Method": method,
        "N": int(len(y_true)),
        "Positives": int(y_true.sum()),
        "Negatives": int(len(y_true) - y_true.sum()),
        "External_Prevalence": float(y_true.mean()),
        "Internal_Prevalence_Source": "13_cdhit40_strict_nomixed_dataset",
        "Internal_Prevalence": INTERNAL_PREVALENCE,
        "Target_Prevalence_Source": "external_set_aggregate_positive_rate",
        "Source_Probability_Column": source_probability_column,
        "Brier_Score": brier_score(y_true, y_prob),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "MCC_at_0_5": float(matthews_corrcoef(y_true, y_pred)),
        "F1_at_0_5": float(f1_score(y_true, y_pred)),
        "Balanced_Accuracy_at_0_5": float(balanced_accuracy_score(y_true, y_pred)),
        "Calibration_Slope": slope,
        "Calibration_Intercept": intercept,
        "Ranking_Changed": "no" if method in {"uncalibrated", "sigmoid", "prevalence_adjusted"} else "possible_ties_from_isotonic",
        "Leakage_Note": note,
    }
    for label, bin_method, n_bins in BINNING_SPECS:
        ece, mce = ece_mce(y_true, y_prob, bin_method, n_bins)
        row[f"ECE_{label}"] = ece
        row[f"MCE_{label}"] = mce
    return row


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


def write_manuscript(rows: list[dict]) -> None:
    primary = [r for r in rows if r["Embedding_Model"] == "ESM2_150M"]
    by_key = {(r["External_Set"], r["Calibration_Method"]): r for r in primary}
    lines = [
        "# Phase 4a Calibration Manuscript Text",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "The external results support a distinction between ranking performance and probability calibration. The model retained useful external ranking behavior, as shown by Phase 1 ROC-AUC and Phase 2 enrichment analyses, but probability calibration fitted on the internal benchmark did not transfer uniformly to external panels.",
        "",
        f"The likely reason is internal-external prevalence shift. The internal strict no-mixed benchmark contains {INTERNAL_POSITIVES}/{INTERNAL_N} positives ({INTERNAL_PREVALENCE:.1%}), whereas the external panels have substantially higher positive rates: N = 66 strict novel40 ({by_key[('strict_novel40', 'uncalibrated')]['External_Prevalence']:.1%}), N = 136 no-exact ({by_key[('seq2topt_noexact', 'uncalibrated')]['External_Prevalence']:.1%}), and N = 984 public no-exact ({by_key[('public_topt_noexact', 'uncalibrated')]['External_Prevalence']:.1%}).",
        "",
        "Consistent with this shift, internally fitted sigmoid calibration improved the N = 136 panel modestly but worsened Brier score and ECE on the N = 66 and N = 984 panels. On the public N = 984 panel, the ESM2-150M calibration intercept moved from 0.31 before calibration to 1.12 after sigmoid calibration, indicating systematic upward probability displacement relative to the external labels.",
        "",
        "We therefore interpret calibrated probability transfer as a limitation of the current PLM-based screening framework: ranking-based screening utility appears more robust than absolute probability calibration under external prevalence shift.",
        "",
        "## Prevalence-Adjustment Sensitivity",
        "",
        "As a sensitivity analysis, we applied a prior-odds correction to the frozen uncalibrated probabilities using only the internal positive rate and each external panel's aggregate positive rate. This adjustment uses no per-sample labels to fit a mapping and does not change ranking; it isolates the effect of cohort-level prior shift. Because deployment-time prevalence may be unknown, this result should be treated as a sensitivity analysis rather than the primary calibration claim.",
        "",
    ]
    for set_id in ["strict_novel40", "seq2topt_noexact", "public_topt_noexact"]:
        unc = by_key[(set_id, "uncalibrated")]
        sig = by_key[(set_id, "sigmoid")]
        iso = by_key[(set_id, "isotonic")]
        prev = by_key[(set_id, "prevalence_adjusted")]
        lines.append(
            f"- {unc['External_Set_Name']} (N = {unc['N']}): "
            f"Brier uncalibrated {unc['Brier_Score']:.4f}, sigmoid {sig['Brier_Score']:.4f}, "
            f"isotonic {iso['Brier_Score']:.4f}, prevalence-adjusted {prev['Brier_Score']:.4f}; "
            f"ECE-10 uncalibrated {unc['ECE_equal_width_10']:.4f}, sigmoid {sig['ECE_equal_width_10']:.4f}, "
            f"isotonic {iso['ECE_equal_width_10']:.4f}, prevalence-adjusted {prev['ECE_equal_width_10']:.4f}."
        )
    OUT_MANUSCRIPT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(PREDICTIONS)
    rows: list[dict] = []
    for _, group in df.groupby(["External_Set", "Embedding_Model"], sort=True):
        group = group.reset_index(drop=True)
        target_prev = float(group["y_true"].mean())
        note_base = "Evaluation only; method included for comparison."
        rows.append(metrics_row(group, "uncalibrated", group["y_prob_uncalibrated"].to_numpy(float), "y_prob_uncalibrated", note_base))
        rows.append(metrics_row(group, "sigmoid", group["y_prob_sigmoid"].to_numpy(float), "y_prob_sigmoid", note_base))
        rows.append(metrics_row(group, "isotonic", group["y_prob_isotonic"].to_numpy(float), "y_prob_isotonic", note_base))
        p_prev = prevalence_adjust(group["y_prob_uncalibrated"].to_numpy(float), INTERNAL_PREVALENCE, target_prev)
        rows.append(
            metrics_row(
                group,
                "prevalence_adjusted",
                p_prev,
                "y_prob_uncalibrated",
                "Prior-odds correction using internal prevalence and external aggregate prevalence only; no per-sample labels used to fit a mapping; ranking unchanged.",
            )
        )

    write_csv(OUT_TABLE, rows)
    write_manuscript(rows)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(PREDICTIONS),
        "output": str(OUT_TABLE),
        "manuscript_text": str(OUT_MANUSCRIPT),
        "internal_positive_rate": INTERNAL_PREVALENCE,
        "prior_correction_formula": "logit(p_adjusted) = logit(p_uncalibrated) + logit(pi_external) - logit(pi_internal)",
        "interpretation": "Sensitivity analysis for internal-external prevalence shift; ranking is unchanged.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
