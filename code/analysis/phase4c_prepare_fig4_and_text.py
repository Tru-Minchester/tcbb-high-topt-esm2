from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
PHASE4 = ROOT / "results" / "analysis" / "phase4"
INTERNAL_CSV = PHASE4 / "fusion_ablation_internal.csv"
EXTERNAL_CSV = PHASE4 / "fusion_ablation_external.csv"
MANIFEST_JSON = PHASE4 / "phase4_ablation_manifest.json"
TEXT_OUT = PHASE4 / "fusion_ablation_manuscript_text.md"
FIG4_LONG_OUT = PHASE4 / "fig4_fusion_ablation_metric_long_table.csv"
FIG4_MANIFEST_OUT = PHASE4 / "phase4c_fig4_manifest.json"

METRICS = ["ROC_AUC", "PR_AUC", "MCC", "F1", "Balanced_Accuracy", "Brier"]


def f(value: float) -> str:
    return f"{float(value):.4f}"


def row_by(df: pd.DataFrame, **kwargs) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in kwargs.items():
        mask &= df[key].astype(str).eq(str(value))
    hits = df[mask]
    if len(hits) != 1:
        raise ValueError(f"Expected one row for {kwargs}; found {len(hits)}")
    return hits.iloc[0]


def make_long_table(internal: pd.DataFrame, external: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for _, r in internal.iterrows():
        for metric in METRICS:
            rows.append(
                {
                    "Figure": "Fig4",
                    "Panel_Suggestion": "internal_controlled_ablation",
                    "Dataset_Scope": "internal",
                    "Dataset_ID": "internal_strict_nomixed_oof",
                    "Dataset_Name": "Internal strict no-mixed 5-fold OOF",
                    "Branch": r["Branch"],
                    "Analysis_Role": r["Analysis_Role"],
                    "Components": r["Components"],
                    "Feature_Dim": r["Feature_Dim"],
                    "Metric": metric,
                    "Value": r[metric],
                    "N": r["N"],
                    "Positives": r["Positives"],
                    "Negatives": r["Negatives"],
                    "Controlled_Ablation_Note": "Same split seed 42, same fixed RF hyperparameters, same SMOTE, same threshold 0.5 across RF branches.",
                }
            )
    for _, r in external.iterrows():
        for metric in METRICS:
            rows.append(
                {
                    "Figure": "Fig4",
                    "Panel_Suggestion": "external_controlled_ablation_heatmap_or_bars",
                    "Dataset_Scope": "external",
                    "Dataset_ID": r["External_Set"],
                    "Dataset_Name": r["External_Set_Name"],
                    "Branch": r["Branch"],
                    "Analysis_Role": r["Analysis_Role"],
                    "Components": r["Components"],
                    "Feature_Dim": r["Feature_Dim"],
                    "Metric": metric,
                    "Value": r[metric],
                    "N": r["N"],
                    "Positives": r["Positives"],
                    "Negatives": r["Negatives"],
                    "Controlled_Ablation_Note": "External sets are evaluation-only; no external labels used for RF hyperparameter selection, late-fusion weights, or calibration.",
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


def write_text(internal: pd.DataFrame, external: pd.DataFrame) -> None:
    dpc_n66 = row_by(external, External_Set="strict_novel40_N66", Branch="DPC")
    dpc_n136 = row_by(external, External_Set="seq2topt_noexact_N136", Branch="DPC")
    esm150_n66 = row_by(external, External_Set="strict_novel40_N66", Branch="ESM2-150M")
    esm150_n136 = row_by(external, External_Set="seq2topt_noexact_N136", Branch="ESM2-150M")
    esm150_n984 = row_by(external, External_Set="public_topt_noexact_N984", Branch="ESM2-150M")

    early150_n66 = row_by(external, External_Set="strict_novel40_N66", Branch="ESM2-150M+handcrafted early fusion")
    early150_n136 = row_by(external, External_Set="seq2topt_noexact_N136", Branch="ESM2-150M+handcrafted early fusion")
    early150_n984 = row_by(external, External_Set="public_topt_noexact_N984", Branch="ESM2-150M+handcrafted early fusion")

    late_internal = row_by(internal, Branch="late fusion")
    calibrated_late_internal = row_by(internal, Branch="calibrated late fusion")
    late_n66 = row_by(external, External_Set="strict_novel40_N66", Branch="late fusion")
    late_n136 = row_by(external, External_Set="seq2topt_noexact_N136", Branch="late fusion")
    late_n984 = row_by(external, External_Set="public_topt_noexact_N984", Branch="late fusion")
    cal_late_n66 = row_by(external, External_Set="strict_novel40_N66", Branch="calibrated late fusion")
    cal_late_n136 = row_by(external, External_Set="seq2topt_noexact_N136", Branch="calibrated late fusion")
    cal_late_n984 = row_by(external, External_Set="public_topt_noexact_N984", Branch="calibrated late fusion")

    lines = [
        "# Phase 4 Controlled Fusion/Ablation Manuscript Text",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Controlled Ablation Protocol",
        "",
        "To isolate feature representation as the only experimental variable, all random-forest branches used the same seed-42 stratified five-fold split, the same SMOTE strategy, the same fixed decision threshold of 0.5, and one fixed RF hyperparameter set selected once using only internal ESM2-150M cross-validation. Therefore, the ESM2-150M values in this controlled ablation may differ slightly from the Phase 1 optimized external-validation model; the purpose here is controlled feature comparison rather than best-model reporting.",
        "",
        "Early fusion concatenated feature vectors before the shared imputer/scaler/SMOTE/RF pipeline. The ESM2-150M early-fusion branch combined 640 ESM2 dimensions with 20 AAC and 400 DPC features (1,060 total dimensions), while the ESM2-650M early-fusion branch combined 1,280 ESM2 dimensions with the same 420 handcrafted features (1,700 total dimensions).",
        "",
        "Late fusion was fitted only on internal out-of-fold probabilities from three base branches: ESM2-150M, AAC, and DPC. It did not include ESM2-650M. The calibrated late-fusion branch applied an internal sigmoid calibrator to the late-fusion OOF score; all external panels were evaluation-only.",
        "",
        "## Main Findings",
        "",
        f"Handcrafted descriptors alone generalized less reliably than PLM embeddings under strict external evaluation. This was most visible for DPC: on the strict novel40 panel, DPC collapsed at the fixed 0.5 threshold with MCC = {f(dpc_n66['MCC'])} and F1 = {f(dpc_n66['F1'])}, and it remained weak on the N = 136 no-exact panel (MCC = {f(dpc_n136['MCC'])}, F1 = {f(dpc_n136['F1'])}). In contrast, ESM2-150M achieved MCC = {f(esm150_n66['MCC'])} and F1 = {f(esm150_n66['F1'])} on novel40, supporting substantially more stable PLM generalization under no-exact-overlap and low-homology external conditions.",
        "",
        f"Early fusion did not provide a stable external gain over ESM2-150M alone. Across the three external panels, ESM2-150M early fusion changed ROC-AUC from {f(esm150_n66['ROC_AUC'])} to {f(early150_n66['ROC_AUC'])} on novel40, from {f(esm150_n136['ROC_AUC'])} to {f(early150_n136['ROC_AUC'])} on N = 136, and from {f(esm150_n984['ROC_AUC'])} to {f(early150_n984['ROC_AUC'])} on N = 984. Thus, two of three external ROC-AUC values decreased slightly. This suggests that ESM2 embeddings already capture much of the discriminative information contained in AAC/DPC, and feature fusion should not be presented as a simple additive improvement.",
        "",
        f"Late fusion produced the lowest internal Brier score among the evaluated branches (Brier = {f(late_internal['Brier'])}; calibrated late fusion = {f(calibrated_late_internal['Brier'])}), but this internal calibration benefit did not translate into uniformly better external ranking. Late fusion had lower ROC-AUC than ESM2-150M on novel40 ({f(late_n66['ROC_AUC'])} vs {f(esm150_n66['ROC_AUC'])}) and N = 136 ({f(late_n136['ROC_AUC'])} vs {f(esm150_n136['ROC_AUC'])}), with only a small increase on N = 984 ({f(late_n984['ROC_AUC'])} vs {f(esm150_n984['ROC_AUC'])}). These results reinforce the broader internal-external shift pattern: internal fusion or calibration gains do not necessarily transfer to external cohorts.",
        "",
        f"Calibrated late fusion had the same external ROC-AUC as uncalibrated late fusion on all three external panels ({f(cal_late_n66['ROC_AUC'])}, {f(cal_late_n136['ROC_AUC'])}, and {f(cal_late_n984['ROC_AUC'])}), as expected for a monotonic sigmoid transformation. Threshold metrics were also unchanged or nearly unchanged, and Brier score did not improve consistently: novel40 {f(late_n66['Brier'])} to {f(cal_late_n66['Brier'])}, N = 136 {f(late_n136['Brier'])} to {f(cal_late_n136['Brier'])}, and N = 984 {f(late_n984['Brier'])} to {f(cal_late_n984['Brier'])}. This branch is therefore not a redundant metric inflation exercise; it verifies that the calibration behavior observed in Phase 3 also holds for the fusion model.",
        "",
        "Overall, the controlled ablation supports PLM embeddings as the main externally robust signal. Handcrafted descriptors are useful references and may help internal or specific external metrics, but neither early fusion nor late fusion produced a consistent external advantage over the PLM single-model branches.",
        "",
    ]
    TEXT_OUT.write_text("\n".join(lines), encoding="utf-8")


def update_manifest() -> None:
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    manifest["phase4c_fig4_and_text"] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manuscript_text": str(TEXT_OUT),
        "fig4_metric_long_table": str(FIG4_LONG_OUT),
        "late_fusion_base_branch_composition": {
            "inputs": ["ESM2-150M", "AAC", "DPC"],
            "input_type": "internal OOF probabilities and corresponding external base-branch probabilities",
            "excludes": ["ESM2-650M"],
            "external_sets": "evaluation_only_no_weight_or_calibrator_fitting",
        },
        "fig4_note": "Long-format table for bar/heatmap plotting; no figure rendered in Phase 4c.",
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    FIG4_MANIFEST_OUT.write_text(json.dumps(manifest["phase4c_fig4_and_text"], ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    internal = pd.read_csv(INTERNAL_CSV)
    external = pd.read_csv(EXTERNAL_CSV)
    rows = make_long_table(internal, external)
    write_csv(FIG4_LONG_OUT, rows)
    write_text(internal, external)
    update_manifest()
    print(
        json.dumps(
            {
                "manuscript_text": str(TEXT_OUT),
                "fig4_metric_long_table": str(FIG4_LONG_OUT),
                "fig4_rows": len(rows),
                "manifest": str(MANIFEST_JSON),
                "phase4c_manifest": str(FIG4_MANIFEST_OUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
