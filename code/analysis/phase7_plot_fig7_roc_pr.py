from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
OUT_DIR = ROOT / "figures" / "phase7_final"

PHASE3_SOURCE = (
    ROOT / "results" / "analysis" / "phase3" / "calibrated_external_predictions.csv"
)
NOVEL40_SOURCE = (
    ROOT / "results"
    / "29_model_scale_external_results.csv"
)
NOVEL40_LEGACY_CROSSCHECK = (
    ROOT / "results"
    / "25_external_test_results.csv"
)
SEQ2TOPT_SOURCE = ROOT / "results" / "32_external_noexact_model_scale_results.csv"
PUBLIC_TOPT_SOURCE = ROOT / "results" / "34_public_topt_noexact_model_scale_results.csv"

STEM = "fig7_roc_pr_external_esm2_150m"
ROC_TOLERANCE = 0.005
PR_TOLERANCE = 0.010
MODEL = "ESM2_150M"

OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
}

COHORTS = [
    {
        "key": "novel40",
        "phase3_key": "strict_novel40",
        "label": "novel40",
        "source": NOVEL40_SOURCE,
        "n": 66,
        "positives": 26,
        "locked_roc_auc": 0.812,
        "locked_pr_auc": 0.745,
        "prevalence": 0.394,
        "color": OKABE_ITO["sky_blue"],
        "linestyle": "-",
    },
    {
        "key": "seq2topt_noexact",
        "phase3_key": "seq2topt_noexact",
        "label": "Seq2Topt no-exact",
        "source": SEQ2TOPT_SOURCE,
        "n": 136,
        "positives": 33,
        "locked_roc_auc": 0.888,
        "locked_pr_auc": 0.752,
        "prevalence": 0.243,
        "color": OKABE_ITO["bluish_green"],
        "linestyle": "--",
    },
    {
        "key": "public_topt_noexact",
        "phase3_key": "public_topt_noexact",
        "label": "Public Topt no-exact",
        "source": PUBLIC_TOPT_SOURCE,
        "n": 984,
        "positives": 313,
        "locked_roc_auc": 0.825,
        "locked_pr_auc": 0.749,
        "prevalence": 0.318,
        "color": OKABE_ITO["orange"],
        "linestyle": "-.",
    },
]


def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_primary(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source)
    if "Embedding_Model" in frame.columns:
        frame = frame.loc[frame["Embedding_Model"].eq(MODEL)].copy()
    required = {"Sequence_Hash", "y_true", "y_prob"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
    frame = frame[["Sequence_Hash", "y_true", "y_prob"]].copy()
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    frame["y_prob"] = pd.to_numeric(frame["y_prob"], errors="raise").astype(float)
    if frame["Sequence_Hash"].duplicated().any():
        raise ValueError(f"Duplicate Sequence_Hash values in {source}")
    if not set(frame["y_true"].unique()).issubset({0, 1}):
        raise ValueError(f"Non-binary y_true values in {source}")
    if not frame["y_prob"].between(0.0, 1.0).all():
        raise ValueError(f"Probabilities outside [0, 1] in {source}")
    return frame


def crosscheck_phase3(
    primary: pd.DataFrame, phase3_all: pd.DataFrame, phase3_key: str
) -> dict[str, float | int | bool]:
    second = phase3_all.loc[
        phase3_all["External_Set"].eq(phase3_key)
        & phase3_all["Embedding_Model"].eq(MODEL),
        ["Sequence_Hash", "y_true", "y_prob_uncalibrated"],
    ].copy()
    if second["Sequence_Hash"].duplicated().any():
        raise ValueError(f"Duplicate Phase 3 hashes for {phase3_key}")
    merged = primary.merge(
        second,
        on="Sequence_Hash",
        how="outer",
        suffixes=("_primary", "_phase3"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = int(merged["_merge"].ne("both").sum())
    comparable = merged.loc[merged["_merge"].eq("both")].copy()
    label_mismatches = int(
        comparable["y_true_primary"].ne(comparable["y_true_phase3"]).sum()
    )
    max_probability_delta = (
        float(
            np.max(
                np.abs(
                    comparable["y_prob"].to_numpy(dtype=float)
                    - comparable["y_prob_uncalibrated"].to_numpy(dtype=float)
                )
            )
        )
        if len(comparable)
        else float("nan")
    )
    passed = (
        unmatched == 0
        and len(comparable) == len(primary) == len(second)
        and label_mismatches == 0
        and max_probability_delta <= 1e-12
    )
    return {
        "phase3_n": int(len(second)),
        "unmatched_hashes": unmatched,
        "label_mismatches": label_mismatches,
        "max_probability_delta": max_probability_delta,
        "phase3_crosscheck_pass": passed,
    }


def crosscheck_novel40_legacy(primary: pd.DataFrame) -> dict[str, float | int | bool]:
    legacy = load_primary(NOVEL40_LEGACY_CROSSCHECK)
    merged = primary.merge(
        legacy,
        on="Sequence_Hash",
        how="outer",
        suffixes=("_primary", "_legacy"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = int(merged["_merge"].ne("both").sum())
    comparable = merged.loc[merged["_merge"].eq("both")]
    label_mismatches = int(
        comparable["y_true_primary"].ne(comparable["y_true_legacy"]).sum()
    )
    max_probability_delta = (
        float(
            np.max(
                np.abs(
                    comparable["y_prob_primary"].to_numpy(dtype=float)
                    - comparable["y_prob_legacy"].to_numpy(dtype=float)
                )
            )
        )
        if len(comparable)
        else float("nan")
    )
    passed = (
        unmatched == 0
        and len(comparable) == len(primary) == len(legacy)
        and label_mismatches == 0
        and max_probability_delta <= 1e-12
    )
    return {
        "legacy_unmatched_hashes": unmatched,
        "legacy_label_mismatches": label_mismatches,
        "legacy_max_probability_delta": max_probability_delta,
        "legacy_crosscheck_pass": passed,
    }


def validate_and_prepare() -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    phase3 = pd.read_csv(PHASE3_SOURCE)
    prepared = []
    manifest_rows = []
    qa_rows = []

    for spec in COHORTS:
        frame = load_primary(spec["source"])
        n = int(len(frame))
        positives = int(frame["y_true"].sum())
        prevalence = positives / n
        if n != spec["n"] or positives != spec["positives"]:
            raise ValueError(
                f"{spec['label']} identity mismatch: got N={n}, positives={positives}; "
                f"expected N={spec['n']}, positives={spec['positives']}"
            )

        phase3_check = crosscheck_phase3(frame, phase3, str(spec["phase3_key"]))
        if not phase3_check["phase3_crosscheck_pass"]:
            raise ValueError(f"Phase 3 cross-check failed for {spec['label']}: {phase3_check}")

        roc_auc = float(roc_auc_score(frame["y_true"], frame["y_prob"]))
        pr_auc = float(average_precision_score(frame["y_true"], frame["y_prob"]))
        fpr, tpr, _ = roc_curve(
            frame["y_true"], frame["y_prob"], drop_intermediate=False
        )
        precision, recall, _ = precision_recall_curve(
            frame["y_true"], frame["y_prob"], drop_intermediate=False
        )
        pr_step_auc = float(-np.sum(np.diff(recall) * precision[:-1]))

        roc_delta = roc_auc - float(spec["locked_roc_auc"])
        pr_delta = pr_auc - float(spec["locked_pr_auc"])
        roc_pass = abs(roc_delta) <= ROC_TOLERANCE
        pr_pass = abs(pr_delta) <= PR_TOLERANCE
        step_pass = abs(pr_step_auc - pr_auc) <= 1e-12
        overall_pass = bool(roc_pass and pr_pass and step_pass)

        if spec["key"] == "novel40":
            legacy_check = crosscheck_novel40_legacy(frame)
            if not legacy_check["legacy_crosscheck_pass"]:
                raise ValueError(f"novel40 legacy cross-check failed: {legacy_check}")
            legacy_source = NOVEL40_LEGACY_CROSSCHECK.relative_to(ROOT).as_posix()
            legacy_sha256 = sha256(NOVEL40_LEGACY_CROSSCHECK)
        else:
            legacy_check = {
                "legacy_unmatched_hashes": np.nan,
                "legacy_label_mismatches": np.nan,
                "legacy_max_probability_delta": np.nan,
                "legacy_crosscheck_pass": np.nan,
            }
            legacy_source = ""
            legacy_sha256 = ""

        manifest_rows.append(
            {
                "Cohort": spec["label"],
                "Primary_Source_File": spec["source"].relative_to(ROOT).as_posix(),
                "Primary_Source_SHA256": sha256(spec["source"]),
                "Primary_Model_Filter": MODEL,
                "Crosscheck_Source_File": PHASE3_SOURCE.relative_to(ROOT).as_posix(),
                "Crosscheck_Source_SHA256": sha256(PHASE3_SOURCE),
                "Legacy_Crosscheck_Source_File": legacy_source,
                "Legacy_Crosscheck_Source_SHA256": legacy_sha256,
                "N": n,
                "Positives": positives,
                "Negatives": n - positives,
                "Prevalence": prevalence,
                **phase3_check,
                **legacy_check,
            }
        )
        qa_rows.append(
            {
                "Cohort": spec["label"],
                "N": n,
                "Positives": positives,
                "Computed_ROC_AUC": roc_auc,
                "Locked_ROC_AUC": spec["locked_roc_auc"],
                "ROC_AUC_Delta": roc_delta,
                "ROC_Tolerance": ROC_TOLERANCE,
                "ROC_PASS": roc_pass,
                "Computed_PR_AUC_Average_Precision": pr_auc,
                "Plotted_PR_Step_Area": pr_step_auc,
                "Locked_PR_AUC": spec["locked_pr_auc"],
                "PR_AUC_Delta": pr_delta,
                "PR_Tolerance": PR_TOLERANCE,
                "PR_PASS": pr_pass,
                "Plotted_Step_Area_Equals_AP": step_pass,
                "Phase3_Crosscheck_PASS": phase3_check["phase3_crosscheck_pass"],
                "Overall_PASS": overall_pass,
            }
        )
        prepared.append(
            {
                **spec,
                "frame": frame,
                "computed_roc_auc": roc_auc,
                "computed_pr_auc": pr_auc,
                "fpr": fpr,
                "tpr": tpr,
                "precision": precision,
                "recall": recall,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    qa = pd.DataFrame(qa_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT_DIR / f"{STEM}_manifest.csv", index=False)
    qa.to_csv(OUT_DIR / f"{STEM}_qa.csv", index=False)
    if not qa["Overall_PASS"].all():
        failed = qa.loc[~qa["Overall_PASS"]]
        raise RuntimeError(
            "Locked AUC validation failed; figure export stopped.\n"
            + failed.to_string(index=False)
        )
    return prepared, manifest, qa


def make_figure(prepared: list[dict[str, object]]) -> plt.Figure:
    set_ieee_style()
    fig, (ax_roc, ax_pr) = plt.subplots(
        1, 2, figsize=(7.16, 3.45), constrained_layout=True
    )

    for item in prepared:
        ax_roc.plot(
            item["fpr"],
            item["tpr"],
            color=item["color"],
            linestyle=item["linestyle"],
            linewidth=1.8,
            label=f"{item['label']} (AUC={item['computed_roc_auc']:.3f})",
            zorder=3,
        )
    ax_roc.plot(
        [0, 1],
        [0, 1],
        color="#777777",
        linewidth=0.9,
        linestyle=(0, (3, 3)),
        label="Chance",
        zorder=1,
    )
    ax_roc.set_title("ROC curves", pad=7)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.legend(loc="lower right", handlelength=2.6)

    for item in prepared:
        ax_pr.axhline(
            float(item["prevalence"]),
            color=item["color"],
            linewidth=0.9,
            linestyle=(0, (1.5, 2.2)),
            zorder=1,
        )
        ax_pr.step(
            item["recall"],
            item["precision"],
            where="post",
            color=item["color"],
            linestyle=item["linestyle"],
            linewidth=1.8,
            label=(
                f"{item['label']} "
                f"(PR-AUC={item['computed_pr_auc']:.3f}; "
                f"prev.={item['prevalence']:.3f})"
            ),
            zorder=3,
        )
    ax_pr.set_title("Precision-recall curves", pad=7)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left", handlelength=2.6)

    for ax, panel_label in [(ax_roc, "A"), (ax_pr, "B")]:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#E6E6E6", linewidth=0.55, zorder=0)
        ax.text(
            -0.14,
            1.04,
            panel_label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    return fig


def save_outputs(fig: plt.Figure) -> list[Path]:
    outputs = []
    for suffix in [".pdf", ".eps"]:
        path = OUT_DIR / f"{STEM}{suffix}"
        fig.savefig(path, facecolor="white")
        outputs.append(path)
    preview = OUT_DIR / f"{STEM}_preview.png"
    fig.savefig(preview, dpi=600, facecolor="white")
    outputs.append(preview)
    return outputs


def main() -> None:
    prepared, manifest, qa = validate_and_prepare()
    fig = make_figure(prepared)
    outputs = save_outputs(fig)
    plt.close(fig)

    summary = qa[
        [
            "Cohort",
            "N",
            "Positives",
            "Computed_ROC_AUC",
            "Locked_ROC_AUC",
            "Computed_PR_AUC_Average_Precision",
            "Locked_PR_AUC",
            "Overall_PASS",
        ]
    ].merge(
        manifest[["Cohort", "Primary_Source_File"]], on="Cohort", how="left"
    )
    summary = summary[
        [
            "Cohort",
            "Primary_Source_File",
            "N",
            "Positives",
            "Computed_ROC_AUC",
            "Locked_ROC_AUC",
            "Computed_PR_AUC_Average_Precision",
            "Locked_PR_AUC",
            "Overall_PASS",
        ]
    ]
    print(summary.to_string(index=False))
    print("\nOutputs:")
    for path in outputs:
        print(f"  {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    for suffix in ["_manifest.csv", "_qa.csv"]:
        path = OUT_DIR / f"{STEM}{suffix}"
        print(f"  {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
