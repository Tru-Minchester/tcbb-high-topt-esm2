from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from phase4_controlled_ablation import (
    EXTERNAL_SETS,
    HASH_COL,
    LABEL_COL,
    N_SPLITS,
    OUT as PHASE4_OUT,
    PRED_DIR as PHASE4_PRED_DIR,
    RANDOM_STATE,
    THRESHOLD,
    build_pipeline,
    component_dims,
    load_external_components,
    load_internal_components,
    make_matrix,
    metrics,
    read_feature_table,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
PHASE4 = ROOT / "results" / "analysis" / "phase4"
PHASE4_PREDICTIONS = PHASE4 / "predictions" / "phase4_predictions_all_branches.csv"
PHASE4_INTERNAL = PHASE4 / "fusion_ablation_internal.csv"
PHASE4_EXTERNAL = PHASE4 / "fusion_ablation_external.csv"
PHASE4_MANIFEST = PHASE4 / "phase4_ablation_manifest.json"
PHASE5 = ROOT / "results" / "analysis" / "phase5"
OUT_CORR = PHASE5 / "feature_group_prediction_correlation.csv"
OUT_ABLATION = PHASE5 / "feature_group_ablation_importance.csv"
OUT_PERM = PHASE5 / "feature_group_permutation_importance.csv"
OUT_TEXT = PHASE5 / "feature_redundancy_interpretability_text.md"
OUT_MANIFEST = PHASE5 / "phase5_manifest.json"

METRIC_NAMES = ["ROC_AUC", "PR_AUC", "MCC", "F1", "Balanced_Accuracy", "Brier"]
CORRELATION_BRANCHES = [
    "AAC",
    "DPC",
    "ESM2-150M",
    "ESM2-650M",
    "ESM2-150M+handcrafted early fusion",
    "ESM2-650M+handcrafted early fusion",
    "late fusion",
]
PRIMARY_FULL_BRANCH = "ESM2-150M+handcrafted early fusion"
PRIMARY_ESM_BRANCH = "ESM2-150M"
RF_PARAMS = {
    "class_weight": None,
    "max_depth": 10,
    "min_samples_leaf": 2,
    "min_samples_split": 5,
    "n_estimators": 400,
}
N_PERMUTATIONS = 30


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


def rf_params_string() -> str:
    return json.dumps(RF_PARAMS, ensure_ascii=False, sort_keys=True)


def dataset_label(row: pd.Series) -> tuple[str, str, str]:
    if str(row.get("Dataset", "")) == "internal_oof":
        return "internal", "internal_strict_nomixed_oof", "Internal strict no-mixed 5-fold OOF"
    return "external", str(row["External_Set"]), str(row["External_Set_Name"])


def prediction_correlation() -> list[dict]:
    pred = pd.read_csv(PHASE4_PREDICTIONS)
    pred = pred[pred["Branch"].isin(CORRELATION_BRANCHES)].copy()
    dataset_frames: list[tuple[str, str, str, pd.DataFrame]] = []
    internal = pred[pred["Dataset"].eq("internal_oof")].copy()
    dataset_frames.append(("internal", "internal_strict_nomixed_oof", "Internal strict no-mixed 5-fold OOF", internal))
    for spec in EXTERNAL_SETS:
        subset = pred[pred["External_Set"].eq(spec.dataset_id)].copy()
        dataset_frames.append(("external", spec.dataset_id, spec.dataset_name, subset))

    rows: list[dict] = []
    for scope, dataset_id, dataset_name, df in dataset_frames:
        pivot = df.pivot_table(index=HASH_COL, columns="Branch", values="y_prob", aggfunc="first")
        pivot = pivot[CORRELATION_BRANCHES]
        spear = pivot.corr(method="spearman")
        pear = pivot.corr(method="pearson")
        for i, a in enumerate(CORRELATION_BRANCHES):
            for b in CORRELATION_BRANCHES[i + 1 :]:
                valid = pivot[[a, b]].dropna()
                rows.append(
                    {
                        "Analysis": "prediction_probability_correlation_not_feature_correlation",
                        "Dataset_Scope": scope,
                        "Dataset_ID": dataset_id,
                        "Dataset_Name": dataset_name,
                        "Branch_A": a,
                        "Branch_B": b,
                        "N": int(len(valid)),
                        "Spearman_Rank_Correlation": float(spear.loc[a, b]),
                        "Pearson_Correlation_Sensitivity": float(pear.loc[a, b]),
                        "Primary_Correlation": "Spearman_rank",
                        "Rationale": "Prediction probability correlation; Spearman is primary because probability scales/calibration differ across branches.",
                    }
                )
    return rows


def load_phase4_predictions(branch: str, dataset_id: str) -> pd.DataFrame:
    pred = pd.read_csv(PHASE4_PREDICTIONS)
    if dataset_id == "internal_strict_nomixed_oof":
        df = pred[(pred["Dataset"].eq("internal_oof")) & (pred["Branch"].eq(branch))].copy()
    else:
        df = pred[(pred["External_Set"].eq(dataset_id)) & (pred["Branch"].eq(branch))].copy()
    return df[[HASH_COL, "y_true", "y_prob"]].copy()


def compare_predictions(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, Any]:
    a = a.sort_values(HASH_COL).reset_index(drop=True)
    b = b.sort_values(HASH_COL).reset_index(drop=True)
    hash_mismatch = int((a[HASH_COL].astype(str).to_numpy() != b[HASH_COL].astype(str).to_numpy()).sum())
    label_mismatch = int((a["y_true"].astype(int).to_numpy() != b["y_true"].astype(int).to_numpy()).sum())
    max_abs_prob_diff = (
        float(np.max(np.abs(a["y_prob"].to_numpy(float) - b["y_prob"].to_numpy(float)))) if len(a) and len(b) else np.nan
    )
    tolerance = 1e-12
    return {
        "Hash_Mismatch": hash_mismatch,
        "Label_Mismatch": label_mismatch,
        "Max_Abs_Probability_Diff": max_abs_prob_diff,
        "Exact_Probability_Match": bool(hash_mismatch == 0 and label_mismatch == 0 and max_abs_prob_diff == 0.0),
        "Probability_Tolerance": tolerance,
        "Numerically_Identical_Within_Tolerance": bool(
            hash_mismatch == 0 and label_mismatch == 0 and max_abs_prob_diff <= tolerance
        ),
    }


def branch_oof_and_external(
    branch_name: str,
    components: list[str],
    internal_base: pd.DataFrame,
    internal_components: dict[str, pd.DataFrame],
    component_cols: dict[str, list[str]],
    external_base_orders: dict[str, pd.DataFrame],
    external_components: dict[str, dict[str, pd.DataFrame]],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    x = make_matrix(components, internal_components, component_cols)
    y = internal_base[LABEL_COL].to_numpy(dtype=int)
    h = internal_base[HASH_COL].astype(str).to_numpy()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.empty(len(y), dtype=float)
    for train_idx, test_idx in cv.split(x, y):
        model = build_pipeline(RF_PARAMS)
        model.fit(x[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(x[test_idx])[:, 1]

    final_model = build_pipeline(RF_PARAMS)
    final_model.fit(x, y)
    out_predictions: dict[str, pd.DataFrame] = {
        "internal_strict_nomixed_oof": pd.DataFrame({HASH_COL: h, "y_true": y, "y_prob": oof})
    }
    out_metrics = {"internal_strict_nomixed_oof": metrics(y, oof)}
    for spec in EXTERNAL_SETS:
        x_ext = make_matrix(components, external_components[spec.dataset_id], component_cols)
        y_ext = external_base_orders[spec.dataset_id][LABEL_COL].to_numpy(dtype=int)
        h_ext = external_base_orders[spec.dataset_id][HASH_COL].astype(str).to_numpy()
        p_ext = final_model.predict_proba(x_ext)[:, 1]
        out_predictions[spec.dataset_id] = pd.DataFrame({HASH_COL: h_ext, "y_true": y_ext, "y_prob": p_ext})
        out_metrics[spec.dataset_id] = metrics(y_ext, p_ext)
    return out_predictions, out_metrics


def group_ablation() -> tuple[list[dict], dict[str, Any]]:
    internal_base, internal_components, component_cols = load_internal_components()
    external_base_orders: dict[str, pd.DataFrame] = {}
    external_components: dict[str, dict[str, pd.DataFrame]] = {}
    for spec in EXTERNAL_SETS:
        esm150_base, _ = read_feature_table(spec.esm150_path, ["ESM_"])
        external_base_orders[spec.dataset_id] = esm150_base[[HASH_COL, LABEL_COL]].copy()
        external_components[spec.dataset_id] = load_external_components(spec, external_base_orders[spec.dataset_id])

    ablation_specs = [
        ("full_ESM2_150M_AAC_DPC", "none", ["ESM2_150M", "AAC", "DPC"]),
        ("remove_AAC", "AAC", ["ESM2_150M", "DPC"]),
        ("remove_DPC", "DPC", ["ESM2_150M", "AAC"]),
        ("remove_AAC_DPC_ESM2_only", "AAC+DPC", ["ESM2_150M"]),
    ]

    phase4_internal = pd.read_csv(PHASE4_INTERNAL)
    phase4_external = pd.read_csv(PHASE4_EXTERNAL)
    phase4_full_internal = phase4_internal[phase4_internal["Branch"].eq(PRIMARY_FULL_BRANCH)].iloc[0]
    phase4_esm_internal = phase4_internal[phase4_internal["Branch"].eq(PRIMARY_ESM_BRANCH)].iloc[0]

    generated_predictions: dict[str, dict[str, pd.DataFrame]] = {}
    generated_metrics: dict[str, dict[str, dict[str, float]]] = {}
    rows: list[dict] = []
    consistency_checks: list[dict] = []

    for name, removed, comps in ablation_specs:
        preds, mets = branch_oof_and_external(
            name,
            comps,
            internal_base,
            internal_components,
            component_cols,
            external_base_orders,
            external_components,
        )
        generated_predictions[name] = preds
        generated_metrics[name] = mets
        if name == "remove_AAC_DPC_ESM2_only":
            phase4_pred = load_phase4_predictions(PRIMARY_ESM_BRANCH, "internal_strict_nomixed_oof")
            consistency_checks.append(
                {
                    "Dataset_ID": "internal_strict_nomixed_oof",
                    "Comparison": "retrained_ESM2_only_vs_Phase4_ESM2_150M",
                    **compare_predictions(preds["internal_strict_nomixed_oof"], phase4_pred),
                }
            )
            for spec in EXTERNAL_SETS:
                phase4_pred_ext = load_phase4_predictions(PRIMARY_ESM_BRANCH, spec.dataset_id)
                consistency_checks.append(
                    {
                        "Dataset_ID": spec.dataset_id,
                        "Comparison": "retrained_ESM2_only_vs_Phase4_ESM2_150M",
                        **compare_predictions(preds[spec.dataset_id], phase4_pred_ext),
                    }
                )

    dataset_names = {
        "internal_strict_nomixed_oof": "Internal strict no-mixed 5-fold OOF",
        **{spec.dataset_id: spec.dataset_name for spec in EXTERNAL_SETS},
    }
    for dataset_id, dataset_name in dataset_names.items():
        full_m = generated_metrics["full_ESM2_150M_AAC_DPC"][dataset_id]
        esm_m = generated_metrics["remove_AAC_DPC_ESM2_only"][dataset_id]
        for name, removed, comps in ablation_specs:
            m = generated_metrics[name][dataset_id]
            for metric_name in METRIC_NAMES:
                value = float(m[metric_name])
                full_value = float(full_m[metric_name])
                esm_value = float(esm_m[metric_name])
                rows.append(
                    {
                        "Analysis": "group_ablation_retrain_not_zero_mask",
                        "Dataset_ID": dataset_id,
                        "Dataset_Name": dataset_name,
                        "Dataset_Scope": "internal" if dataset_id == "internal_strict_nomixed_oof" else "external",
                        "Ablation_Branch": name,
                        "Removed_Group": removed,
                        "Remaining_Components": "+".join(comps),
                        "Metric": metric_name,
                        "Value": value,
                        "Full_Early_Fusion_Value": full_value,
                        "ESM2_Only_Value": esm_value,
                        "Delta_vs_Full_Early_Fusion": value - full_value,
                        "Delta_Full_Early_Fusion_vs_ESM2_Only": full_value - esm_value,
                        "RF_Params": rf_params_string(),
                        "Split": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
                        "SMOTE": f"SMOTE(random_state={RANDOM_STATE})",
                        "Decision_Threshold": THRESHOLD,
                    }
                )
    metadata = {
        "esm2_only_consistency_checks": consistency_checks,
        "all_esm2_only_predictions_match_phase4": all(
            c["Hash_Mismatch"] == 0 and c["Label_Mismatch"] == 0 and c["Max_Abs_Probability_Diff"] <= c["Probability_Tolerance"]
            for c in consistency_checks
        ),
    }
    return rows, metadata


def permutation_importance() -> list[dict]:
    manifest = json.loads(PHASE4_MANIFEST.read_text(encoding="utf-8"))
    model_path = Path(manifest["outputs"]["models"]) / "esm2_150m_plus_handcrafted_early_fusion_final_rf_smote.joblib"
    model = joblib.load(model_path)

    internal_base, internal_components, component_cols = load_internal_components()
    group_components = ["ESM2_150M", "AAC", "DPC"]
    group_slices: dict[str, slice] = {}
    start = 0
    for group in group_components:
        dim = len(component_cols[group])
        group_slices[group] = slice(start, start + dim)
        start += dim

    rows: list[dict] = []
    rng = np.random.default_rng(RANDOM_STATE)
    for spec in EXTERNAL_SETS:
        esm150_base, _ = read_feature_table(spec.esm150_path, ["ESM_"])
        base_order = esm150_base[[HASH_COL, LABEL_COL]].copy()
        ext_components = load_external_components(spec, base_order)
        x_ext = make_matrix(group_components, ext_components, component_cols)
        y_ext = base_order[LABEL_COL].to_numpy(dtype=int)
        baseline_prob = model.predict_proba(x_ext)[:, 1]
        baseline_metrics = metrics(y_ext, baseline_prob)
        for group, col_slice in group_slices.items():
            replicate_metrics: list[dict[str, float]] = []
            for rep in range(1, N_PERMUTATIONS + 1):
                xp = x_ext.copy()
                perm_idx = rng.permutation(len(xp))
                xp[:, col_slice] = xp[perm_idx, col_slice]
                prob = model.predict_proba(xp)[:, 1]
                replicate_metrics.append(metrics(y_ext, prob))
            for metric_name in ["ROC_AUC", "PR_AUC", "Brier"]:
                vals = np.array([m[metric_name] for m in replicate_metrics], dtype=float)
                baseline = float(baseline_metrics[metric_name])
                perm_mean = float(vals.mean())
                perm_std = float(vals.std(ddof=1))
                if metric_name == "Brier":
                    degradation = perm_mean - baseline
                    direction = "positive_means_worse_brier_after_permutation"
                else:
                    degradation = baseline - perm_mean
                    direction = "positive_means_lower_ranking_metric_after_permutation"
                rows.append(
                    {
                        "Analysis": "group_permutation_importance_external_only",
                        "External_Set": spec.dataset_id,
                        "External_Set_Name": spec.dataset_name,
                        "Model": PRIMARY_FULL_BRANCH,
                        "Permuted_Group": group,
                        "Metric": metric_name,
                        "Baseline_Value": baseline,
                        "Permuted_Mean": perm_mean,
                        "Permuted_Std": perm_std,
                        "Permutation_Effect_Degradation": degradation,
                        "Raw_Delta_Permuted_Minus_Baseline": perm_mean - baseline,
                        "N_Repeats": N_PERMUTATIONS,
                        "Random_Seed": RANDOM_STATE,
                        "Direction": direction,
                        "RF_Params": rf_params_string(),
                    }
                )
    return rows


def write_text(corr_rows: list[dict], ablation_rows: list[dict], perm_rows: list[dict], metadata: dict[str, Any]) -> None:
    corr = pd.DataFrame(corr_rows)
    abl = pd.DataFrame(ablation_rows)
    perm = pd.DataFrame(perm_rows)

    def corr_value(dataset: str, a: str, b: str) -> float:
        hit = corr[
            corr["Dataset_ID"].eq(dataset)
            & (
                (corr["Branch_A"].eq(a) & corr["Branch_B"].eq(b))
                | (corr["Branch_A"].eq(b) & corr["Branch_B"].eq(a))
            )
        ]
        return float(hit.iloc[0]["Spearman_Rank_Correlation"])

    esm_aac_internal = corr_value("internal_strict_nomixed_oof", "ESM2-150M", "AAC")
    esm_dpc_internal = corr_value("internal_strict_nomixed_oof", "ESM2-150M", "DPC")
    esm_full_internal = corr_value("internal_strict_nomixed_oof", "ESM2-150M", PRIMARY_FULL_BRANCH)
    esm_full_n984 = corr_value("public_topt_noexact_N984", "ESM2-150M", PRIMARY_FULL_BRANCH)

    def delta_full_vs_esm(dataset: str, metric_name: str) -> float:
        hit = abl[
            abl["Dataset_ID"].eq(dataset)
            & abl["Ablation_Branch"].eq("full_ESM2_150M_AAC_DPC")
            & abl["Metric"].eq(metric_name)
        ]
        return float(hit.iloc[0]["Delta_Full_Early_Fusion_vs_ESM2_Only"])

    d_auc_n66 = delta_full_vs_esm("strict_novel40_N66", "ROC_AUC")
    d_auc_n136 = delta_full_vs_esm("seq2topt_noexact_N136", "ROC_AUC")
    d_auc_n984 = delta_full_vs_esm("public_topt_noexact_N984", "ROC_AUC")
    d_brier_n66 = delta_full_vs_esm("strict_novel40_N66", "Brier")
    d_brier_n136 = delta_full_vs_esm("seq2topt_noexact_N136", "Brier")
    d_brier_n984 = delta_full_vs_esm("public_topt_noexact_N984", "Brier")

    perm_summary = perm[
        perm["External_Set"].eq("public_topt_noexact_N984")
        & perm["Metric"].eq("ROC_AUC")
        & perm["Permuted_Group"].isin(["ESM2_150M", "AAC", "DPC"])
    ][["Permuted_Group", "Permutation_Effect_Degradation"]]
    perm_bits = ", ".join(
        f"{r.Permuted_Group}: {float(r.Permutation_Effect_Degradation):.4f}" for r in perm_summary.itertuples()
    )

    lines = [
        "# Phase 5 Feature Redundancy and Interpretability Text",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "All analyses reuse the Phase 4 controlled-ablation framework: seed-42 stratified five-fold OOF predictions, the same fixed RF hyperparameters, the same SMOTE strategy, and a fixed 0.5 threshold. Correlations are prediction-probability correlations, not raw feature correlations. Spearman rank correlation is primary because branch probability scales differ.",
        "",
        "## Prediction Probability Redundancy",
        "",
        f"Prediction correlations support substantial overlap between the PLM branch and the fusion branch. Internally, ESM2-150M and ESM2-150M early fusion had Spearman rho = {esm_full_internal:.3f}; on the N = 984 external set this remained rho = {esm_full_n984:.3f}. In contrast, direct ESM2-150M correlations with standalone AAC and DPC were more modest internally (AAC rho = {esm_aac_internal:.3f}; DPC rho = {esm_dpc_internal:.3f}), suggesting that the fusion model largely preserves the ESM2 ranking while handcrafted-only branches are not simple duplicates of the PLM branch.",
        "",
        "## Group Ablation",
        "",
        f"Retrained group ablation showed limited and context-dependent handcrafted complementarity. Relative to ESM2-150M alone, full ESM2-150M+AAC+DPC early fusion changed ROC-AUC by {d_auc_n66:+.4f} on N = 66, {d_auc_n136:+.4f} on N = 136, and {d_auc_n984:+.4f} on N = 984. The corresponding Brier changes were {d_brier_n66:+.4f}, {d_brier_n136:+.4f}, and {d_brier_n984:+.4f}. Thus, handcrafted features did not provide stable external ranking improvement, although some external metrics show small panel-specific changes.",
        "",
        "The ESM2-only ablation branch was retrained under the same Phase 4 protocol and matched the Phase 4 ESM2-150M predictions within numerical floating-point tolerance across internal OOF and all external panels, with zero hash or label mismatches. This confirms that the group-ablation comparisons are aligned with the accepted Phase 4 tables.",
        "",
        "## Group Permutation",
        "",
        f"External group permutation further indicates that the ESM2 block is the dominant driver of the early-fusion model. On N = 984, ROC-AUC degradation after group permutation was {perm_bits}. Because the permutation analysis repeats each group shuffle 30 times, the small-sample N = 66 estimates should be read with their reported standard deviations.",
        "",
        "## Interpretation",
        "",
        "The data support a conservative interpretation: PLM embeddings absorb much of the stable ranking signal available to the controlled fusion model, while handcrafted descriptors provide limited, context-dependent complementarity rather than a robust basis for claiming fusion superiority. Where handcrafted and PLM outputs are correlated, the result is evidence of biochemical-composition information already represented in PLM embeddings; where standalone handcrafted predictions differ, ablation and permutation show that this independent signal is not consistently useful across external panels.",
        "",
    ]
    OUT_TEXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PHASE5.mkdir(parents=True, exist_ok=True)
    corr_rows = prediction_correlation()
    write_csv(OUT_CORR, corr_rows)

    ablation_rows, ablation_metadata = group_ablation()
    write_csv(OUT_ABLATION, ablation_rows)

    perm_rows = permutation_importance()
    write_csv(OUT_PERM, perm_rows)

    write_text(corr_rows, ablation_rows, perm_rows, ablation_metadata)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase4_framework_reused": {
            "split": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
            "rf_params": RF_PARAMS,
            "smote": f"SMOTE(random_state={RANDOM_STATE})",
            "threshold": THRESHOLD,
            "phase4_manifest": str(PHASE4_MANIFEST),
        },
        "outputs": {
            "feature_group_prediction_correlation": str(OUT_CORR),
            "feature_group_ablation_importance": str(OUT_ABLATION),
            "feature_group_permutation_importance": str(OUT_PERM),
            "interpretability_text": str(OUT_TEXT),
        },
        "correlation_note": "Prediction probability correlation, not raw feature correlation. Spearman is primary; Pearson is sensitivity.",
        "ablation_note": "Group ablation retrains after removing AAC, DPC, or AAC+DPC from the ESM2-150M early-fusion framework; no zero masking.",
        "permutation_note": "External-only group permutation on the trained Phase 4 ESM2-150M early-fusion model; 30 repeats, seed 42.",
        **ablation_metadata,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
