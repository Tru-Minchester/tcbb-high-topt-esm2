from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
RELEASE = ROOT
FEATURES = RELEASE / "features"
RESULTS = RELEASE / "results"
PHASE1_MAIN = ROOT / "results" / "analysis" / "phase1" / "external_validation_main_table.csv"
PHASE2_BOOTSTRAP = ROOT / "results" / "analysis" / "phase2" / "screening_bootstrap_ci.csv"
OUT = ROOT / "results" / "analysis" / "phase3"
CALIBRATOR_DIR = OUT / "models" / "calibrators"

RANDOM_STATE = 42
CALIBRATION_SEED = 20260713
N_SPLITS = 5
LABEL_COL = "Binary_Label"
HASH_COL = "Sequence_Hash"
PROB_CLIP = 1e-6


@dataclass(frozen=True)
class ModelSpec:
    model: str
    feature_file: str
    feature_prefix: str
    analysis_role: str


@dataclass(frozen=True)
class ExternalSet:
    set_id: str
    display_name: str
    file_name: str
    n_expected: int


MODELS = [
    ModelSpec("ESM2_150M", "15_strict_nomixed_esm2_features.csv", "ESM_", "primary"),
    ModelSpec("ESM2_650M", "27_strict_nomixed_esm2_650m_features.csv", "ESM650_", "model_scale_sensitivity"),
]

EXTERNAL_SETS = [
    ExternalSet("strict_novel40", "Strict Seq2Topt novel40 external", "29_model_scale_external_results.csv", 66),
    ExternalSet("seq2topt_noexact", "Seq2Topt no-exact-overlap external", "32_external_noexact_model_scale_results.csv", 136),
    ExternalSet("public_topt_noexact", "Public Topt no-exact-overlap external", "34_public_topt_noexact_model_scale_results.csv", 984),
]

BINNING_SPECS = [
    ("equal_width_10", "equal_width", 10, "primary"),
    ("equal_width_5", "equal_width", 5, "small_sample_sensitivity"),
    ("quantile_5", "quantile", 5, "small_sample_sensitivity"),
]


def logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p.astype(float), PROB_CLIP, 1.0 - PROB_CLIP)
    return np.log(clipped / (1.0 - clipped))


def parse_best_params(raw: str) -> dict[str, Any]:
    params = json.loads(raw)
    clean: dict[str, Any] = {}
    for key, value in params.items():
        clean[key.replace("clf__", "")] = value
    return clean


def load_internal_features(spec: ModelSpec) -> tuple[pd.DataFrame, list[str]]:
    path = FEATURES / spec.feature_file
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith(spec.feature_prefix)]
    missing = [c for c in [HASH_COL, LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    if not feat_cols:
        raise ValueError(f"No feature columns found for {spec.model} in {path}")
    df = df.dropna(subset=[HASH_COL, LABEL_COL]).drop_duplicates(subset=[HASH_COL]).copy()
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").astype(int)
    return df, feat_cols


def build_pipeline(params: dict[str, Any]) -> ImbPipeline:
    clf_params = {
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        **params,
    }
    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(**clf_params)),
        ]
    )


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob.astype(float) - y_true.astype(float)) ** 2))


def classification_snapshot(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Brier_Score": brier_score(y_true, y_prob),
    }


def make_oof_predictions(spec: ModelSpec) -> tuple[pd.DataFrame, list[dict]]:
    df, feat_cols = load_internal_features(spec)
    fold_results = pd.read_csv(RESULTS / "29_model_scale_internal_fold_results.csv")
    fold_results = fold_results[fold_results["Embedding_Model"] == spec.model].sort_values("Fold").reset_index(drop=True)
    if len(fold_results) != N_SPLITS:
        raise ValueError(f"Expected {N_SPLITS} frozen internal fold rows for {spec.model}; found {len(fold_results)}")

    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=int)
    hashes = df[HASH_COL].astype(str).to_numpy()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_rows: list[dict] = []
    check_rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        frozen_row = fold_results[fold_results["Fold"] == fold].iloc[0]
        params = parse_best_params(str(frozen_row["Best_Params"]))
        pipe = build_pipeline(params)
        pipe.fit(X[train_idx], y[train_idx])
        y_prob = pipe.predict_proba(X[test_idx])[:, 1]
        metrics = classification_snapshot(y[test_idx], y_prob)
        check_row = {
            "Embedding_Model": spec.model,
            "Fold": fold,
            "N_Test": int(len(test_idx)),
            "Pos_Test": int(y[test_idx].sum()),
            "Best_Params_Reused": json.dumps(params, ensure_ascii=False, sort_keys=True),
        }
        for metric in ["ROC_AUC", "PR_AUC", "MCC", "F1", "Balanced_Accuracy"]:
            observed = metrics[metric]
            frozen = float(frozen_row[metric])
            check_row[f"{metric}_Recomputed"] = observed
            check_row[f"{metric}_Frozen_Internal_Fold"] = frozen
            check_row[f"{metric}_Abs_Diff"] = abs(observed - frozen)
        check_rows.append(check_row)
        for idx, prob in zip(test_idx, y_prob):
            oof_rows.append(
                {
                    "Sequence_Hash": hashes[idx],
                    "Embedding_Model": spec.model,
                    "Fold": fold,
                    "y_true": int(y[idx]),
                    "y_prob_oof_uncalibrated": float(prob),
                }
            )

    return pd.DataFrame(oof_rows), check_rows


def fit_calibrators(oof_df: pd.DataFrame, model: str) -> dict[str, Any]:
    y = oof_df["y_true"].to_numpy(dtype=int)
    p = oof_df["y_prob_oof_uncalibrated"].to_numpy(dtype=float)
    x = logit(p).reshape(-1, 1)

    sigmoid = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000, random_state=CALIBRATION_SEED)
    sigmoid.fit(x, y)

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p, y)

    joblib.dump(sigmoid, CALIBRATOR_DIR / f"{model.lower()}_sigmoid_calibrator.pkl")
    joblib.dump(isotonic, CALIBRATOR_DIR / f"{model.lower()}_isotonic_calibrator.pkl")

    metadata = {
        "model": model,
        "sigmoid_logit_intercept": float(sigmoid.intercept_[0]),
        "sigmoid_logit_coef": float(sigmoid.coef_[0][0]),
        "sigmoid_probability_clip": PROB_CLIP,
        "isotonic_threshold_count": int(len(isotonic.X_thresholds_)),
        "isotonic_x_thresholds": [float(x) for x in isotonic.X_thresholds_],
        "isotonic_y_thresholds": [float(x) for x in isotonic.y_thresholds_],
    }
    (CALIBRATOR_DIR / f"{model.lower()}_calibrator_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"sigmoid": sigmoid, "isotonic": isotonic, "metadata": metadata}


def apply_calibrator(method: str, p: np.ndarray, calibrators: dict[str, Any]) -> np.ndarray:
    if method == "uncalibrated":
        return p.astype(float)
    if method == "sigmoid":
        return calibrators["sigmoid"].predict_proba(logit(p).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return np.clip(calibrators["isotonic"].predict(p.astype(float)), 0.0, 1.0)
    raise ValueError(f"Unknown calibration method: {method}")


def equal_width_rows(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n:
            mean_pred = float(y_prob[mask].mean())
            obs = float(y_true[mask].mean())
            gap = abs(obs - mean_pred)
        else:
            mean_pred = math.nan
            obs = math.nan
            gap = math.nan
        rows.append(
            {
                "Bin_Index": i + 1,
                "Bin_Lower": lo,
                "Bin_Upper": hi,
                "Bin_Closed": "right" if i == n_bins - 1 else "left",
                "N_Bin": n,
                "Mean_Predicted_Probability": mean_pred,
                "Observed_Positive_Fraction": obs,
                "Absolute_Calibration_Gap": gap,
            }
        )
    return rows


def quantile_rows(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> list[dict]:
    order = np.argsort(y_prob, kind="mergesort")
    chunks = np.array_split(order, n_bins)
    rows = []
    for i, idx in enumerate(chunks):
        n = int(len(idx))
        if n:
            preds = y_prob[idx]
            obs_y = y_true[idx]
            mean_pred = float(preds.mean())
            obs = float(obs_y.mean())
            lo = float(preds.min())
            hi = float(preds.max())
            gap = abs(obs - mean_pred)
        else:
            mean_pred = math.nan
            obs = math.nan
            lo = math.nan
            hi = math.nan
            gap = math.nan
        rows.append(
            {
                "Bin_Index": i + 1,
                "Bin_Lower": lo,
                "Bin_Upper": hi,
                "Bin_Closed": "rank_chunk",
                "N_Bin": n,
                "Mean_Predicted_Probability": mean_pred,
                "Observed_Positive_Fraction": obs,
                "Absolute_Calibration_Gap": gap,
            }
        )
    return rows


def reliability_rows(y_true: np.ndarray, y_prob: np.ndarray, binning_method: str, n_bins: int) -> list[dict]:
    if binning_method == "equal_width":
        return equal_width_rows(y_true, y_prob, n_bins)
    if binning_method == "quantile":
        return quantile_rows(y_true, y_prob, n_bins)
    raise ValueError(f"Unknown binning method: {binning_method}")


def ece_mce(rows: list[dict], n_total: int) -> tuple[float, float]:
    nonempty = [r for r in rows if int(r["N_Bin"]) > 0]
    ece = sum((int(r["N_Bin"]) / n_total) * float(r["Absolute_Calibration_Gap"]) for r in nonempty)
    mce = max((float(r["Absolute_Calibration_Gap"]) for r in nonempty), default=math.nan)
    return float(ece), float(mce)


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, str]:
    if len(np.unique(y_true)) < 2:
        return math.nan, math.nan, "not_estimable_single_class"
    x = logit(y_prob).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000, random_state=CALIBRATION_SEED)
    model.fit(x, y_true.astype(int))
    return float(model.coef_[0][0]), float(model.intercept_[0]), "logistic_regression_on_logit_probability_C_1e6"


def load_external_set(spec: ExternalSet) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / spec.file_name)
    if "Embedding_Model" not in df.columns:
        df["Embedding_Model"] = "ESM2_150M"
    if len(df) % spec.n_expected != 0:
        raise ValueError(f"{spec.file_name} row count {len(df)} incompatible with expected N={spec.n_expected}")
    df["External_Set"] = spec.set_id
    df["External_Set_Name"] = spec.display_name
    df["y_true"] = df["y_true"].astype(int)
    df["y_prob"] = df["y_prob"].astype(float)
    return df


def phase1_brier_lookup() -> dict[tuple[str, str], float]:
    phase1 = pd.read_csv(PHASE1_MAIN)
    lookup = {}
    for _, row in phase1.iterrows():
        lookup[(str(row["External_Set"]), str(row["Embedding_Model"]))] = float(row["Brier_Score"])
    return lookup


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


def write_screening_effect() -> None:
    if not PHASE2_BOOTSTRAP.exists():
        return
    phase2 = pd.read_csv(PHASE2_BOOTSTRAP)
    keep = phase2[phase2["Metric"].isin(["Precision_TopK", "Recall_TopK", "EF_Observed"])].copy()
    rows: list[dict] = []
    for method in ["uncalibrated", "sigmoid"]:
        for _, row in keep.iterrows():
            item = row.to_dict()
            item["Calibration_Method"] = method
            item["Ranking_Effect"] = (
                "baseline_ranking"
                if method == "uncalibrated"
                else "unchanged_by_strictly_monotonic_sigmoid_calibration"
            )
            item["Interpretation"] = "Calibration affects probability reliability; sigmoid calibration does not change TopK ranking."
            rows.append(item)
    write_csv(OUT / "calibration_screening_effect.csv", rows)


def write_manuscript_text(metric_rows: list[dict], brier_checks: list[dict]) -> None:
    primary = [
        r
        for r in metric_rows
        if r["Embedding_Model"] == "ESM2_150M"
        and r["Calibration_Method"] in {"uncalibrated", "sigmoid"}
        and r["Dataset_Type"] == "external"
    ]
    lines = [
        "# Phase 3 Calibration Manuscript Text",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Calibration was fitted exclusively on internal strict no-mixed cross-fitted predictions. No external validation set was used for fitting the calibrator, selecting the calibration method, or tuning probability bins. The uncalibrated external probabilities were reused from the frozen Phase 1/2 prediction files, and their Brier scores were reconciled against Phase 1 before calibration.",
        "",
        "Platt/sigmoid calibration was used as the primary calibration method, with isotonic regression reported as a sensitivity analysis. Reliability diagrams used 10 equal-width probability bins as the primary binning rule. For the smaller N = 66 and N = 136 external sets, 5-bin equal-width and 5-bin quantile ECE were included as sensitivity analyses; small-sample ECE should be interpreted as auxiliary evidence, with calibration slope and intercept emphasized for the N = 66 panel.",
        "",
        "Because sigmoid calibration is monotonic, it is not expected to improve ranking-based screening metrics such as EF, Precision@TopK, or Recall@TopK. The calibration analysis evaluates probability reliability for wet-lab decision thresholds rather than ranking enrichment, and calibration should not be described as beneficial unless the external Brier/ECE results support that claim.",
        "",
        "## Primary ESM2-150M External Calibration Summary",
        "",
    ]
    by_key = {(r["External_Set"], r["Calibration_Method"]): r for r in primary}
    for set_id in ["strict_novel40", "seq2topt_noexact", "public_topt_noexact"]:
        unc = by_key.get((set_id, "uncalibrated"))
        sig = by_key.get((set_id, "sigmoid"))
        if unc is None or sig is None:
            continue
        lines.append(
            f"- {unc['External_Set_Name']} (N = {unc['N']}; positives = {unc['Positives']}, negatives = {unc['Negatives']}): "
            f"Brier {unc['Brier_Score']:.4f} to {sig['Brier_Score']:.4f}; "
            f"ECE-10 {unc['ECE_equal_width_10']:.4f} to {sig['ECE_equal_width_10']:.4f}; "
            f"calibration slope {unc['Calibration_Slope']:.3f} to {sig['Calibration_Slope']:.3f}; "
            f"intercept {unc['Calibration_Intercept']:.3f} to {sig['Calibration_Intercept']:.3f}."
        )
    lines += [
        "",
        "Overall, sigmoid calibration showed mixed external behavior rather than uniform improvement. The manuscript should report calibrated probabilities as a prespecified calibration analysis, not claim that calibration improved every external panel.",
    ]
    max_diff = max((abs(float(r["Brier_Diff_vs_Phase1"])) for r in brier_checks), default=math.nan)
    lines += [
        "",
        "## Frozen Probability Reconciliation",
        "",
        f"The maximum absolute difference between Phase 3 uncalibrated Brier scores and Phase 1 Brier scores was {max_diff:.12g}.",
        "",
    ]
    (OUT / "calibration_manuscript_text.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CALIBRATOR_DIR.mkdir(parents=True, exist_ok=True)

    all_oof: list[pd.DataFrame] = []
    internal_check_rows: list[dict] = []
    calibrators_by_model: dict[str, dict[str, Any]] = {}
    for spec in MODELS:
        oof_df, check_rows = make_oof_predictions(spec)
        all_oof.append(oof_df)
        internal_check_rows.extend(check_rows)
        calibrators_by_model[spec.model] = fit_calibrators(oof_df, spec.model)

    oof_all = pd.concat(all_oof, ignore_index=True)
    oof_all.to_csv(OUT / "internal_oof_uncalibrated_predictions.csv", index=False)
    write_csv(OUT / "internal_oof_reproduction_checks.csv", internal_check_rows)

    phase1_brier = phase1_brier_lookup()
    metric_rows: list[dict] = []
    reliability_out: list[dict] = []
    prediction_rows: list[dict] = []
    brier_checks: list[dict] = []

    external_all = pd.concat([load_external_set(spec) for spec in EXTERNAL_SETS], ignore_index=True)
    spec_by_set = {spec.set_id: spec for spec in EXTERNAL_SETS}
    model_role = {spec.model: spec.analysis_role for spec in MODELS}

    for (external_set, model), group in external_all.groupby(["External_Set", "Embedding_Model"], sort=True):
        group = group.reset_index(drop=True)
        y = group["y_true"].to_numpy(dtype=int)
        p_unc = group["y_prob"].to_numpy(dtype=float)
        brier = brier_score(y, p_unc)
        phase1_value = phase1_brier[(external_set, model)]
        brier_checks.append(
            {
                "External_Set": external_set,
                "Embedding_Model": model,
                "Phase3_Uncalibrated_Brier": brier,
                "Phase1_Brier": phase1_value,
                "Brier_Diff_vs_Phase1": brier - phase1_value,
                "Brier_Matches_Phase1": abs(brier - phase1_value) <= 1e-12,
                "Frozen_Uncalibrated_Source": str(RESULTS / spec_by_set[external_set].file_name),
            }
        )

        probs_by_method = {
            method: apply_calibrator(method, p_unc, calibrators_by_model[model])
            for method in ["uncalibrated", "sigmoid", "isotonic"]
        }
        for i, row in group.iterrows():
            prediction_rows.append(
                {
                    "External_Set": external_set,
                    "External_Set_Name": row["External_Set_Name"],
                    "Embedding_Model": model,
                    "Sequence_Hash": row.get("Sequence_Hash", ""),
                    "y_true": int(row["y_true"]),
                    "y_prob_uncalibrated": float(probs_by_method["uncalibrated"][i]),
                    "y_prob_sigmoid": float(probs_by_method["sigmoid"][i]),
                    "y_prob_isotonic": float(probs_by_method["isotonic"][i]),
                    "Frozen_Uncalibrated_Source": str(RESULTS / spec_by_set[external_set].file_name),
                }
            )

        for method, p in probs_by_method.items():
            snapshot = classification_snapshot(y, p)
            slope, intercept, slope_method = calibration_slope_intercept(y, p)
            row_metrics: dict[str, Any] = {
                "Dataset_Type": "external",
                "External_Set": external_set,
                "External_Set_Name": group["External_Set_Name"].iloc[0],
                "Embedding_Model": model,
                "Analysis_Role": model_role.get(model, ""),
                "Calibration_Method": method,
                "N": int(len(y)),
                "Positives": int(y.sum()),
                "Negatives": int(len(y) - y.sum()),
                "Positive_Rate": float(y.mean()),
                "Brier_Score": snapshot["Brier_Score"],
                "ROC_AUC": snapshot["ROC_AUC"],
                "PR_AUC": snapshot["PR_AUC"],
                "MCC_at_0_5": snapshot["MCC"],
                "F1_at_0_5": snapshot["F1"],
                "Balanced_Accuracy_at_0_5": snapshot["Balanced_Accuracy"],
                "Calibration_Slope": slope,
                "Calibration_Intercept": intercept,
                "Calibration_Slope_Intercept_Method": slope_method,
                "Small_Sample_ECE_Note": "small_sample_ece_auxiliary" if len(y) <= 136 else "",
                "Frozen_Uncalibrated_Source": str(RESULTS / spec_by_set[external_set].file_name),
            }
            for bin_label, binning_method, n_bins, bin_role in BINNING_SPECS:
                rows = reliability_rows(y, p, binning_method, n_bins)
                ece, mce = ece_mce(rows, len(y))
                row_metrics[f"ECE_{bin_label}"] = ece
                row_metrics[f"MCE_{bin_label}"] = mce
                for r in rows:
                    r_out = {
                        "Dataset_Type": "external",
                        "External_Set": external_set,
                        "External_Set_Name": group["External_Set_Name"].iloc[0],
                        "Embedding_Model": model,
                        "Analysis_Role": model_role.get(model, ""),
                        "Calibration_Method": method,
                        "Binning_Label": bin_label,
                        "Binning_Method": binning_method,
                        "N_Bins": n_bins,
                        "Binning_Role": bin_role,
                        "N_Total": int(len(y)),
                        "Positives": int(y.sum()),
                        "Negatives": int(len(y) - y.sum()),
                        **r,
                    }
                    reliability_out.append(r_out)
            metric_rows.append(row_metrics)

    write_csv(OUT / "calibration_metrics.csv", metric_rows)
    write_csv(OUT / "reliability_diagram_data.csv", reliability_out)
    write_csv(OUT / "calibrated_external_predictions.csv", prediction_rows)
    write_csv(OUT / "external_brier_reconciliation_with_phase1.csv", brier_checks)
    write_screening_effect()
    write_manuscript_text(metric_rows, brier_checks)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "random_state_base_model": RANDOM_STATE,
        "calibration_seed": CALIBRATION_SEED,
        "n_internal_crossfit_folds": N_SPLITS,
        "base_model_consistency_rule": "External uncalibrated probabilities are read directly from the frozen Phase 1/2 prediction files; they are not regenerated in Phase 3.",
        "external_brier_reconciliation": brier_checks,
        "internal_oof_generation": "RF/SMOTE pipeline refit within frozen Phase 29 internal folds using stored fold-level best parameters to generate internal out-of-fold probabilities for calibrator fitting.",
        "calibration_methods": ["uncalibrated", "sigmoid_primary", "isotonic_sensitivity"],
        "binning_specs": [
            {"label": label, "method": method, "n_bins": n_bins, "role": role}
            for label, method, n_bins, role in BINNING_SPECS
        ],
        "brier_definition": "mean((predicted_probability - observed_label)^2)",
        "calibration_slope_intercept": "Logistic regression of observed label on logit(predicted_probability), ideal slope=1 and intercept=0.",
        "small_sample_ece_note": "For N=66 and N=136, ECE is auxiliary; equal_width_5 and quantile_5 are sensitivity binning schemes.",
        "outputs": {
            "calibration_metrics": str(OUT / "calibration_metrics.csv"),
            "reliability_diagram_data": str(OUT / "reliability_diagram_data.csv"),
            "calibrated_external_predictions": str(OUT / "calibrated_external_predictions.csv"),
            "internal_oof_uncalibrated_predictions": str(OUT / "internal_oof_uncalibrated_predictions.csv"),
            "internal_oof_reproduction_checks": str(OUT / "internal_oof_reproduction_checks.csv"),
            "external_brier_reconciliation_with_phase1": str(OUT / "external_brier_reconciliation_with_phase1.csv"),
            "calibration_screening_effect": str(OUT / "calibration_screening_effect.csv"),
            "calibration_manuscript_text": str(OUT / "calibration_manuscript_text.md"),
            "calibrators": str(CALIBRATOR_DIR),
        },
    }
    (OUT / "calibration_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
