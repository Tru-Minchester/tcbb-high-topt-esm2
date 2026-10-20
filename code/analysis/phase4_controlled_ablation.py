from __future__ import annotations

import csv
import json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
RELEASE = ROOT
FEATURES = RELEASE / "features"
PHASE4_FEATURES = ROOT / "results" / "analysis" / "phase4_feature_matrices"
OUT = ROOT / "results" / "analysis" / "phase4"
MODELS_DIR = OUT / "models"
PRED_DIR = OUT / "predictions"

HASH_COL = "Sequence_Hash"
LABEL_COL = "Binary_Label"
THRESHOLD = 0.5
RANDOM_STATE = 42
N_SPLITS = 5
PROB_CLIP = 1e-6


@dataclass(frozen=True)
class ExternalSet:
    dataset_id: str
    dataset_name: str
    expected_n: int
    aac_path: Path
    dpc_path: Path
    esm150_path: Path
    esm650_path: Path


EXTERNAL_SETS = [
    ExternalSet(
        "strict_novel40_N66",
        "Strict Seq2Topt novel40 external",
        66,
        PHASE4_FEATURES / "strict_novel40_N66_aac_features.csv",
        PHASE4_FEATURES / "strict_novel40_N66_dpc_features.csv",
        FEATURES / "24_external_esm2_features.csv",
        FEATURES / "28_external_esm2_650m_features.csv",
    ),
    ExternalSet(
        "seq2topt_noexact_N136",
        "Seq2Topt no-exact-overlap external",
        136,
        PHASE4_FEATURES / "seq2topt_noexact_N136_aac_features.csv",
        PHASE4_FEATURES / "seq2topt_noexact_N136_dpc_features.csv",
        FEATURES / "32_external_noexact_esm2_150m_features.csv",
        FEATURES / "32_external_noexact_esm2_650m_features.csv",
    ),
    ExternalSet(
        "public_topt_noexact_N984",
        "Public Topt no-exact-overlap external",
        984,
        PHASE4_FEATURES / "public_topt_noexact_N984_aac_features.csv",
        PHASE4_FEATURES / "public_topt_noexact_N984_dpc_features.csv",
        FEATURES / "34_public_topt_noexact_esm2_150m_features.csv",
        FEATURES / "34_public_topt_noexact_esm2_650m_features.csv",
    ),
]


FEATURE_SOURCES = {
    "AAC": {
        "internal": FEATURES / "14_strict_nomixed_aac_features.csv",
        "prefixes": ["AAC_"],
        "component_dims": {"AAC": 20},
    },
    "DPC": {
        "internal": FEATURES / "14_strict_nomixed_dpc_features.csv",
        "prefixes": ["DPC_"],
        "component_dims": {"DPC": 400},
    },
    "ESM2_150M": {
        "internal": FEATURES / "15_strict_nomixed_esm2_features.csv",
        "prefixes": ["ESM_"],
        "component_dims": {"ESM2_150M": 640},
    },
    "ESM2_650M": {
        "internal": FEATURES / "27_strict_nomixed_esm2_650m_features.csv",
        "prefixes": ["ESM650_"],
        "component_dims": {"ESM2_650M": 1280},
    },
}


BRANCH_SPECS = [
    {"branch": "AAC", "components": ["AAC"], "analysis_role": "handcrafted"},
    {"branch": "DPC", "components": ["DPC"], "analysis_role": "handcrafted"},
    {"branch": "AAC+DPC", "components": ["AAC", "DPC"], "analysis_role": "handcrafted"},
    {"branch": "ESM2-150M", "components": ["ESM2_150M"], "analysis_role": "primary_plm"},
    {"branch": "ESM2-650M", "components": ["ESM2_650M"], "analysis_role": "model_scale_sensitivity"},
    {
        "branch": "ESM2-150M+handcrafted early fusion",
        "components": ["ESM2_150M", "AAC", "DPC"],
        "analysis_role": "primary_early_fusion",
    },
    {
        "branch": "ESM2-650M+handcrafted early fusion",
        "components": ["ESM2_650M", "AAC", "DPC"],
        "analysis_role": "model_scale_sensitivity_early_fusion",
    },
]


RF_SELECTION_GRID = {
    "clf__n_estimators": [400],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2],
    "clf__class_weight": [None, "balanced_subsample"],
}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), PROB_CLIP, 1.0 - PROB_CLIP)
    return np.log(p / (1.0 - p))


def build_pipeline(rf_params: dict[str, Any]) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **rf_params)),
        ]
    )


def feature_columns(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    cols = [c for c in df.columns if any(c.startswith(prefix) for prefix in prefixes)]
    if not cols:
        raise ValueError(f"No feature columns for prefixes={prefixes}")
    return cols


def read_feature_table(path: Path, prefixes: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    if HASH_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"{path} missing {HASH_COL}/{LABEL_COL}")
    df[HASH_COL] = df[HASH_COL].astype(str)
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").astype(int)
    cols = feature_columns(df, prefixes)
    keep = [HASH_COL, LABEL_COL] + cols
    return df[keep].drop_duplicates(subset=[HASH_COL]).copy(), cols


def align_by_hash(base: pd.DataFrame, other: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = set(base[HASH_COL]) - set(other[HASH_COL])
    extra = set(other[HASH_COL]) - set(base[HASH_COL])
    if missing or extra:
        raise ValueError(f"{name} hash set mismatch: missing={len(missing)}, extra={len(extra)}")
    aligned = other.set_index(HASH_COL).loc[base[HASH_COL].astype(str)].reset_index()
    label_mismatch = int((base[LABEL_COL].astype(int).to_numpy() != aligned[LABEL_COL].astype(int).to_numpy()).sum())
    if label_mismatch:
        raise ValueError(f"{name} label mismatch after alignment: {label_mismatch}")
    return aligned


def load_internal_components() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, list[str]]]:
    component_tables: dict[str, pd.DataFrame] = {}
    component_cols: dict[str, list[str]] = {}
    for component, spec in FEATURE_SOURCES.items():
        df, cols = read_feature_table(spec["internal"], spec["prefixes"])
        component_tables[component] = df
        component_cols[component] = cols
    base = component_tables["ESM2_150M"][[HASH_COL, LABEL_COL]].copy()
    for component, df in list(component_tables.items()):
        component_tables[component] = align_by_hash(base, df, f"internal_{component}")
    return base, component_tables, component_cols


def load_external_components(spec: ExternalSet, base_order: pd.DataFrame) -> dict[str, pd.DataFrame]:
    paths = {
        "AAC": spec.aac_path,
        "DPC": spec.dpc_path,
        "ESM2_150M": spec.esm150_path,
        "ESM2_650M": spec.esm650_path,
    }
    external: dict[str, pd.DataFrame] = {}
    for component, path in paths.items():
        df, _ = read_feature_table(path, FEATURE_SOURCES[component]["prefixes"])
        external[component] = align_by_hash(base_order, df, f"{spec.dataset_id}_{component}")
    return external


def make_matrix(
    components: list[str],
    component_tables: dict[str, pd.DataFrame],
    component_cols: dict[str, list[str]],
) -> np.ndarray:
    matrices = [component_tables[c][component_cols[c]].to_numpy(dtype=np.float32) for c in components]
    return np.concatenate(matrices, axis=1) if len(matrices) > 1 else matrices[0]


def component_dims(components: list[str], component_cols: dict[str, list[str]]) -> dict[str, int]:
    return {component: len(component_cols[component]) for component in components}


def metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= THRESHOLD).astype(int)
    return {
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Brier": float(np.mean((y_prob.astype(float) - y_true.astype(float)) ** 2)),
    }


def format_params(params: dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def select_rf_params(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=build_pipeline({}),
        param_grid=RF_SELECTION_GRID,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    grid.fit(x, y)
    results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score").reset_index(drop=True)
    selected = {key.replace("clf__", ""): value for key, value in grid.best_params_.items()}
    return selected, results


def run_branch(
    branch_spec: dict,
    internal_base: pd.DataFrame,
    internal_components: dict[str, pd.DataFrame],
    component_cols: dict[str, list[str]],
    external_base_orders: dict[str, pd.DataFrame],
    external_components: dict[str, dict[str, pd.DataFrame]],
    rf_params: dict[str, Any],
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    branch = branch_spec["branch"]
    components = branch_spec["components"]
    role = branch_spec["analysis_role"]
    x = make_matrix(components, internal_components, component_cols)
    y = internal_base[LABEL_COL].to_numpy(dtype=int)
    hashes = internal_base[HASH_COL].astype(str).to_numpy()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.empty(len(y), dtype=float)
    internal_pred_rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        model = build_pipeline(rf_params)
        model.fit(x[train_idx], y[train_idx])
        fold_prob = model.predict_proba(x[test_idx])[:, 1]
        oof[test_idx] = fold_prob
        for idx, prob in zip(test_idx, fold_prob):
            internal_pred_rows.append(
                {
                    "Dataset": "internal_oof",
                    "Branch": branch,
                    "Analysis_Role": role,
                    "Fold": fold,
                    "Sequence_Hash": hashes[idx],
                    "y_true": int(y[idx]),
                    "y_prob": float(prob),
                }
            )

    final_model = build_pipeline(rf_params)
    final_model.fit(x, y)
    joblib.dump(final_model, MODELS_DIR / f"{safe_name(branch)}_final_rf_smote.joblib")

    internal_row = {
        "Dataset": "internal_strict_nomixed_oof",
        "Branch": branch,
        "Analysis_Role": role,
        "Components": "+".join(components),
        "Feature_Dim": int(x.shape[1]),
        "Component_Dims": json.dumps(component_dims(components, component_cols), sort_keys=True),
        "N": int(len(y)),
        "Positives": int(y.sum()),
        "Negatives": int(len(y) - y.sum()),
        "RF_Params": format_params(rf_params),
        "SMOTE": f"SMOTE(random_state={RANDOM_STATE})",
        "CV": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        "Decision_Threshold": THRESHOLD,
        **metrics(y, oof),
    }

    external_rows: list[dict] = []
    external_pred_rows: list[dict] = []
    for ext_spec in EXTERNAL_SETS:
        ext_components = external_components[ext_spec.dataset_id]
        x_ext = make_matrix(components, ext_components, component_cols)
        y_ext = external_base_orders[ext_spec.dataset_id][LABEL_COL].to_numpy(dtype=int)
        h_ext = external_base_orders[ext_spec.dataset_id][HASH_COL].astype(str).to_numpy()
        prob = final_model.predict_proba(x_ext)[:, 1]
        external_rows.append(
            {
                "External_Set": ext_spec.dataset_id,
                "External_Set_Name": ext_spec.dataset_name,
                "Branch": branch,
                "Analysis_Role": role,
                "Components": "+".join(components),
                "Feature_Dim": int(x_ext.shape[1]),
                "Component_Dims": json.dumps(component_dims(components, component_cols), sort_keys=True),
                "N": int(len(y_ext)),
                "Positives": int(y_ext.sum()),
                "Negatives": int(len(y_ext) - y_ext.sum()),
                "RF_Params": format_params(rf_params),
                "SMOTE": f"SMOTE(random_state={RANDOM_STATE})",
                "Training": "final_model_fit_on_full_internal_strict_nomixed",
                "Decision_Threshold": THRESHOLD,
                **metrics(y_ext, prob),
            }
        )
        for h, yt, p in zip(h_ext, y_ext, prob):
            external_pred_rows.append(
                {
                    "External_Set": ext_spec.dataset_id,
                    "External_Set_Name": ext_spec.dataset_name,
                    "Branch": branch,
                    "Analysis_Role": role,
                    "Sequence_Hash": h,
                    "y_true": int(yt),
                    "y_prob": float(p),
                }
            )

    oof_info = {
        "branch": branch,
        "role": role,
        "components": components,
        "feature_dim": int(x.shape[1]),
        "oof_prob": oof,
    }
    return oof_info, [internal_row], external_rows, internal_pred_rows + external_pred_rows


def safe_name(name: str) -> str:
    return (
        name.replace("+", "_plus_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
        .lower()
    )


def fit_late_fusion(
    oof_by_branch: dict[str, dict],
    internal_base: pd.DataFrame,
    external_base_orders: dict[str, pd.DataFrame],
    external_pred_by_branch_set: dict[tuple[str, str], pd.DataFrame],
) -> tuple[list[dict], list[dict], list[dict], dict]:
    y = internal_base[LABEL_COL].to_numpy(dtype=int)
    hashes = internal_base[HASH_COL].astype(str).to_numpy()
    branch_inputs = ["ESM2-150M", "AAC", "DPC"]
    x_meta = np.vstack([oof_by_branch[b]["oof_prob"] for b in branch_inputs]).T
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    late_oof = np.empty(len(y), dtype=float)
    calibrated_oof = np.empty(len(y), dtype=float)
    for fold, (train_idx, test_idx) in enumerate(cv.split(x_meta, y), start=1):
        meta = LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)
        meta.fit(x_meta[train_idx], y[train_idx])
        late_oof[test_idx] = meta.predict_proba(x_meta[test_idx])[:, 1]

    x_cal = logit(late_oof).reshape(-1, 1)
    for fold, (train_idx, test_idx) in enumerate(cv.split(x_cal, y), start=1):
        cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000, random_state=RANDOM_STATE)
        cal.fit(x_cal[train_idx], y[train_idx])
        calibrated_oof[test_idx] = cal.predict_proba(x_cal[test_idx])[:, 1]

    final_meta = LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)
    final_meta.fit(x_meta, y)
    final_cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000, random_state=RANDOM_STATE)
    final_cal.fit(logit(late_oof).reshape(-1, 1), y)
    joblib.dump(final_meta, MODELS_DIR / "late_fusion_meta_learner.joblib")
    joblib.dump(final_cal, MODELS_DIR / "calibrated_late_fusion_sigmoid.joblib")

    internal_rows: list[dict] = []
    pred_rows: list[dict] = []
    for branch, prob, role in [
        ("late fusion", late_oof, "primary_late_fusion"),
        ("calibrated late fusion", calibrated_oof, "primary_calibrated_late_fusion"),
    ]:
        internal_rows.append(
            {
                "Dataset": "internal_strict_nomixed_oof",
                "Branch": branch,
                "Analysis_Role": role,
                "Components": "+".join(branch_inputs),
                "Feature_Dim": len(branch_inputs),
                "Component_Dims": json.dumps({"meta_input_probabilities": len(branch_inputs)}, sort_keys=True),
                "N": int(len(y)),
                "Positives": int(y.sum()),
                "Negatives": int(len(y) - y.sum()),
                "RF_Params": "base_branch_fixed_rf_params; meta_learner_logistic_regression_internal_oof_only",
                "SMOTE": "applied_only_inside_base_branches",
                "CV": f"base OOF + meta/calibration StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
                "Decision_Threshold": THRESHOLD,
                **metrics(y, prob),
            }
        )
        for h, yt, p in zip(hashes, y, prob):
            pred_rows.append(
                {
                    "Dataset": "internal_oof",
                    "Branch": branch,
                    "Analysis_Role": role,
                    "Fold": "",
                    "Sequence_Hash": h,
                    "y_true": int(yt),
                    "y_prob": float(p),
                }
            )

    external_rows: list[dict] = []
    for ext_spec in EXTERNAL_SETS:
        ext_y = external_base_orders[ext_spec.dataset_id][LABEL_COL].to_numpy(dtype=int)
        ext_h = external_base_orders[ext_spec.dataset_id][HASH_COL].astype(str).to_numpy()
        ext_meta = np.vstack(
            [
                external_pred_by_branch_set[(b, ext_spec.dataset_id)]["y_prob"].to_numpy(dtype=float)
                for b in branch_inputs
            ]
        ).T
        late_prob = final_meta.predict_proba(ext_meta)[:, 1]
        cal_prob = final_cal.predict_proba(logit(late_prob).reshape(-1, 1))[:, 1]
        for branch, prob, role in [
            ("late fusion", late_prob, "primary_late_fusion"),
            ("calibrated late fusion", cal_prob, "primary_calibrated_late_fusion"),
        ]:
            external_rows.append(
                {
                    "External_Set": ext_spec.dataset_id,
                    "External_Set_Name": ext_spec.dataset_name,
                    "Branch": branch,
                    "Analysis_Role": role,
                    "Components": "+".join(branch_inputs),
                    "Feature_Dim": len(branch_inputs),
                    "Component_Dims": json.dumps({"meta_input_probabilities": len(branch_inputs)}, sort_keys=True),
                    "N": int(len(ext_y)),
                    "Positives": int(ext_y.sum()),
                    "Negatives": int(len(ext_y) - ext_y.sum()),
                    "RF_Params": "base_branch_fixed_rf_params; meta_learner_logistic_regression_internal_oof_only",
                    "SMOTE": "applied_only_inside_base_branches",
                    "Training": "meta_learner_fit_on_internal_oof_base_probabilities; external_evaluation_only",
                    "Decision_Threshold": THRESHOLD,
                    **metrics(ext_y, prob),
                }
            )
            for h, yt, p in zip(ext_h, ext_y, prob):
                pred_rows.append(
                    {
                        "External_Set": ext_spec.dataset_id,
                        "External_Set_Name": ext_spec.dataset_name,
                        "Branch": branch,
                        "Analysis_Role": role,
                        "Sequence_Hash": h,
                        "y_true": int(yt),
                        "y_prob": float(p),
                    }
                )

    meta_info = {
        "late_fusion_inputs": branch_inputs,
        "meta_learner": "LogisticRegression(max_iter=10000, random_state=42)",
        "calibrated_late_fusion": "Sigmoid/Platt LogisticRegression on internal late-fusion OOF logits only",
        "meta_coefficients": final_meta.coef_[0].round(10).tolist(),
        "meta_intercept": float(final_meta.intercept_[0]),
        "calibrator_coefficient": float(final_cal.coef_[0][0]),
        "calibrator_intercept": float(final_cal.intercept_[0]),
    }
    return internal_rows, external_rows, pred_rows, meta_info


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    internal_base, internal_components, component_cols = load_internal_components()
    y = internal_base[LABEL_COL].to_numpy(dtype=int)
    x_select = make_matrix(["ESM2_150M"], internal_components, component_cols)
    selected_rf, selection_results = select_rf_params(x_select, y)
    selection_results.to_csv(OUT / "rf_hyperparameter_selection_esm2_150m_seed42.csv", index=False)

    external_base_orders: dict[str, pd.DataFrame] = {}
    external_components: dict[str, dict[str, pd.DataFrame]] = {}
    for spec in EXTERNAL_SETS:
        esm150_base, _ = read_feature_table(spec.esm150_path, ["ESM_"])
        external_base_orders[spec.dataset_id] = esm150_base[[HASH_COL, LABEL_COL]].copy()
        external_components[spec.dataset_id] = load_external_components(spec, external_base_orders[spec.dataset_id])

    internal_rows: list[dict] = []
    external_rows: list[dict] = []
    all_prediction_rows: list[dict] = []
    oof_by_branch: dict[str, dict] = {}
    external_pred_by_branch_set: dict[tuple[str, str], pd.DataFrame] = {}
    branch_param_checks: list[dict] = []

    for branch_spec in BRANCH_SPECS:
        oof_info, int_rows, ext_rows, pred_rows = run_branch(
            branch_spec,
            internal_base,
            internal_components,
            component_cols,
            external_base_orders,
            external_components,
            selected_rf,
        )
        oof_by_branch[branch_spec["branch"]] = oof_info
        internal_rows.extend(int_rows)
        external_rows.extend(ext_rows)
        all_prediction_rows.extend(pred_rows)
        branch_param_checks.append(
            {
                "Branch": branch_spec["branch"],
                "RF_Params": format_params(selected_rf),
                "Matches_Selected_RF_Params": True,
                "SMOTE": f"SMOTE(random_state={RANDOM_STATE})",
                "Decision_Threshold": THRESHOLD,
            }
        )

    external_base_preds = [r for r in all_prediction_rows if "External_Set" in r]
    pred_df = pd.DataFrame(external_base_preds)
    for branch in [b["branch"] for b in BRANCH_SPECS]:
        for ext_spec in EXTERNAL_SETS:
            external_pred_by_branch_set[(branch, ext_spec.dataset_id)] = pred_df[
                (pred_df["Branch"] == branch) & (pred_df["External_Set"] == ext_spec.dataset_id)
            ][[HASH_COL, "y_true", "y_prob"]].copy()

    late_internal, late_external, late_predictions, meta_info = fit_late_fusion(
        oof_by_branch,
        internal_base,
        external_base_orders,
        external_pred_by_branch_set,
    )
    internal_rows.extend(late_internal)
    external_rows.extend(late_external)
    all_prediction_rows.extend(late_predictions)

    write_csv(OUT / "fusion_ablation_internal.csv", internal_rows)
    write_csv(OUT / "fusion_ablation_external.csv", external_rows)
    write_csv(OUT / "phase4_branch_hyperparam_checks.csv", branch_param_checks)
    write_csv(PRED_DIR / "phase4_predictions_all_branches.csv", all_prediction_rows)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "Phase 4 controlled ablation: only feature representation changes across RF branches.",
        "split": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        "random_state": RANDOM_STATE,
        "threshold": THRESHOLD,
        "smote": f"SMOTE(random_state={RANDOM_STATE})",
        "rf_hyperparameter_selection": {
            "selection_branch": "ESM2-150M",
            "selection_data": "internal strict no-mixed only",
            "selection_cv": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
            "selection_metric": "ROC_AUC",
            "candidate_grid": RF_SELECTION_GRID,
            "selected_rf_params": selected_rf,
            "selection_results": str(OUT / "rf_hyperparameter_selection_esm2_150m_seed42.csv"),
        },
        "controlled_ablation_note": "All RF branches use the same selected RF hyperparameters, same SMOTE strategy, same split seed, and fixed threshold 0.5. ESM2 branch values may differ from Phase 1 optimized models because this is a controlled ablation rather than best-model reporting.",
        "branch_component_dimensions": {
            branch_spec["branch"]: {
                "components": branch_spec["components"],
                "component_dims": component_dims(branch_spec["components"], component_cols),
                "total_dim": sum(component_dims(branch_spec["components"], component_cols).values()),
            }
            for branch_spec in BRANCH_SPECS
        },
        "late_fusion": meta_info,
        "hyperparameter_check": {
            "file": str(OUT / "phase4_branch_hyperparam_checks.csv"),
            "all_rf_branch_params_identical": all(r["Matches_Selected_RF_Params"] for r in branch_param_checks),
        },
        "outputs": {
            "fusion_ablation_internal": str(OUT / "fusion_ablation_internal.csv"),
            "fusion_ablation_external": str(OUT / "fusion_ablation_external.csv"),
            "predictions_all_branches": str(PRED_DIR / "phase4_predictions_all_branches.csv"),
            "models": str(MODELS_DIR),
        },
    }
    (OUT / "phase4_ablation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
