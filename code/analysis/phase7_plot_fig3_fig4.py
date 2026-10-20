from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
OUT_DIR = ROOT / "figures" / "phase7_final"

PHASE1 = ROOT / "results" / "analysis" / "phase1"
PHASE4 = ROOT / "results" / "analysis" / "phase4"
PHASE5 = ROOT / "results" / "analysis" / "phase5"

FIG3_SOURCE = PHASE1 / "external_validation_main_table.csv"
FIG4_HEATMAP_SOURCE = PHASE4 / "fig4_fusion_ablation_metric_long_table.csv"
FIG4_CORR_SOURCE = PHASE5 / "feature_group_prediction_correlation.csv"
FIG4_PERM_SOURCE = PHASE5 / "feature_group_permutation_importance.csv"


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

DATASET_COLORS = {
    "strict_novel40": OKABE_ITO["sky_blue"],
    "seq2topt_noexact": OKABE_ITO["bluish_green"],
    "public_topt_noexact": OKABE_ITO["orange"],
}

MODEL_COLORS = {
    "ESM2-150M": OKABE_ITO["blue"],
    "AAC": OKABE_ITO["orange"],
    "DPC": OKABE_ITO["bluish_green"],
}

FIG3_CONFIG_LABEL = "ESM2-150M: optimized external-validation model"
FIG3_FOOTNOTE = (
    "Note: The ESM2-150M column in Fig 4A is from a fixed-hyperparameter controlled ablation\n"
    "and differs from these optimized values by ≤0.045 across all metrics and panels."
)
FIG4A_CONFIG_LABEL = "All branches: fixed-RF controlled ablation (feature comparison only)"
FIG4_FOOTNOTE = (
    "Note: To isolate feature representation, all branches use one fixed RF hyperparameter set.\n"
    "ESM2-150M values differ by ≤0.045 from the optimized external-validation model in Fig 3."
)

DATASET_ORDER = ["strict_novel40", "seq2topt_noexact", "public_topt_noexact"]
DATASET_LABELS = {
    "strict_novel40": "N=66\nnovel40",
    "seq2topt_noexact": "N=136\nno-exact",
    "public_topt_noexact": "N=984\nno-exact",
    "strict_novel40_N66": "N=66",
    "seq2topt_noexact_N136": "N=136",
    "public_topt_noexact_N984": "N=984",
    "internal_strict_nomixed_oof": "Internal",
}

BRANCH_ORDER = [
    "AAC",
    "DPC",
    "AAC+DPC",
    "ESM2-150M",
    "ESM2-150M+handcrafted early fusion",
    "late fusion",
    "calibrated late fusion",
]
BRANCH_LABELS = {
    "AAC": "AAC",
    "DPC": "DPC",
    "AAC+DPC": "AAC+DPC",
    "ESM2-150M": "ESM2",
    "ESM2-150M+handcrafted early fusion": "Early",
    "late fusion": "Late",
    "calibrated late fusion": "Cal. late",
}


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
            "legend.fontsize": 8,
            "figure.titlesize": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def parse_ci(ci_text: str) -> tuple[float, float]:
    if pd.isna(ci_text):
        return (math.nan, math.nan)
    lo, hi = str(ci_text).split("-")
    return float(lo), float(hi)


def save_all(fig: plt.Figure, stem: str, raster_dpi: int = 600) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in [".pdf", ".eps"]:
        path = OUT_DIR / f"{stem}{suffix}"
        fig.savefig(path)
        outputs.append(path)
    png_path = OUT_DIR / f"{stem}_preview.png"
    fig.savefig(png_path, dpi=raster_dpi)
    outputs.append(png_path)
    return outputs


def make_fig3() -> dict[str, object]:
    df = pd.read_csv(FIG3_SOURCE)
    df = df[df["Embedding_Model"].eq("ESM2_150M")].copy()
    df["External_Set"] = pd.Categorical(df["External_Set"], DATASET_ORDER, ordered=True)
    df = df.sort_values("External_Set")

    metric_specs = [
        ("ROC_AUC", "ROC-AUC", "ROC_AUC_CI95", "higher"),
        ("PR_AUC", "PR-AUC", "PR_AUC_CI95", "higher"),
        ("MCC", "MCC", "MCC_CI95", "higher"),
        ("Brier_Score", "Brier score", "Brier_Score_CI95", "lower"),
    ]
    colors = [DATASET_COLORS[str(ds)] for ds in df["External_Set"].astype(str)]
    hatches = ["", "//", "\\\\"]

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.35), constrained_layout=True)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.135, 1.0, 0.94))
    axes = axes.ravel()
    qa_rows = []

    for ax, (metric, label, ci_col, direction) in zip(axes, metric_specs):
        values = df[metric].to_numpy(dtype=float)
        lows, highs = zip(*(parse_ci(x) for x in df[ci_col]))
        lows = np.asarray(lows, dtype=float)
        highs = np.asarray(highs, dtype=float)
        err_low = values - lows
        err_high = highs - values
        if np.any(lows < -1e-12):
            raise ValueError(f"{metric} CI lower bound below zero: {lows}")
        if np.any(err_low < -1e-12) or np.any(err_high < -1e-12):
            raise ValueError(f"{metric} has negative error bar components.")

        x = np.arange(len(df))
        bars = ax.bar(
            x,
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            yerr=np.vstack([err_low, err_high]),
            error_kw={"elinewidth": 0.8, "capsize": 3, "capthick": 0.8},
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[v] for v in df["External_Set"].astype(str)])
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.set_axisbelow(True)
        if metric == "MCC":
            ax.set_ylim(0, min(1.0, max(highs) + 0.12))
        else:
            ax.set_ylim(0, min(1.0, max(highs) + 0.08))
        if direction == "lower":
            ax.text(0.98, 0.95, "lower is better", ha="right", va="top", transform=ax.transAxes, fontsize=8)

        for _, row in df.iterrows():
            lo, hi = parse_ci(row[ci_col])
            qa_rows.append(
                {
                    "Figure": "Fig3",
                    "Metric": metric,
                    "Dataset": row["External_Set"],
                    "Value": row[metric],
                    "CI_Lower": lo,
                    "CI_Upper": hi,
                    "Error_Bar_Lower_Nonnegative": bool(lo >= -1e-12),
                    "Model_Filter": "ESM2_150M only",
                }
            )

    fig.text(0.5, 1.12, FIG3_CONFIG_LABEL, ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.text(0.5, 0.025, FIG3_FOOTNOTE, ha="center", va="bottom", fontsize=7, fontstyle="italic")
    outputs = save_all(fig, "fig3_external_validation_esm2_150m")
    plt.close(fig)
    return {"outputs": outputs, "qa": qa_rows}


def heatmap_panel(ax: plt.Axes, matrix: pd.DataFrame, title: str, cmap: str, vmin: float, vmax: float) -> matplotlib.image.AxesImage:
    arr = matrix.to_numpy(dtype=float)
    im = ax.imshow(arr, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax), aspect="auto")
    ax.set_title(title, pad=3)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=0)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                scaled = (val - vmin) / max(vmax - vmin, 1e-9)
                bright_for_good = scaled > 0.60 if cmap != "cividis_r" else scaled < 0.40
                txt_color = "white" if bright_for_good else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_color)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def make_fig4() -> dict[str, object]:
    long_df = pd.read_csv(FIG4_HEATMAP_SOURCE)
    corr_df = pd.read_csv(FIG4_CORR_SOURCE)
    perm_df = pd.read_csv(FIG4_PERM_SOURCE)

    branch_filter = [b for b in BRANCH_ORDER if b in set(long_df["Branch"])]
    dataset_order = [
        "internal_strict_nomixed_oof",
        "strict_novel40_N66",
        "seq2topt_noexact_N136",
        "public_topt_noexact_N984",
    ]
    dataset_labels = [DATASET_LABELS[d] for d in dataset_order]
    metric_order = ["ROC_AUC", "PR_AUC", "MCC", "F1", "Balanced_Accuracy", "Brier"]
    metric_titles = {
        "ROC_AUC": "ROC-AUC",
        "PR_AUC": "PR-AUC",
        "MCC": "MCC",
        "F1": "F1",
        "Balanced_Accuracy": "Balanced accuracy",
        "Brier": "Brier",
    }

    fig = plt.figure(figsize=(7.16, 10.6), constrained_layout=False)
    outer = gridspec.GridSpec(3, 1, height_ratios=[2.45, 1.45, 1.25], hspace=0.38, figure=fig)
    heat_gs = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=outer[0], wspace=0.22, hspace=0.33)
    corr_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[1.05, 0.08], wspace=0.05)
    perm_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.22)

    heat_axes = []
    for idx, metric in enumerate(metric_order):
        ax = fig.add_subplot(heat_gs[idx // 2, idx % 2])
        sub = long_df[
            long_df["Branch"].isin(branch_filter)
            & long_df["Dataset_ID"].isin(dataset_order)
            & long_df["Metric"].eq(metric)
        ].copy()
        sub["Branch"] = pd.Categorical(sub["Branch"], branch_filter, ordered=True)
        sub["Dataset_ID"] = pd.Categorical(sub["Dataset_ID"], dataset_order, ordered=True)
        pivot = sub.pivot(index="Branch", columns="Dataset_ID", values="Value").sort_index()
        pivot = pivot.reindex(columns=dataset_order)
        pivot.index = [BRANCH_LABELS.get(x, x) for x in pivot.index.astype(str)]
        pivot.columns = dataset_labels
        cmap = "cividis_r" if metric == "Brier" else "cividis"
        vmin = float(np.nanmin(pivot.to_numpy(dtype=float)))
        vmax = float(np.nanmax(pivot.to_numpy(dtype=float)))
        title = metric_titles[metric] if metric != "Brier" else "Brier\nlower is better"
        im = heatmap_panel(ax, pivot, title, cmap, vmin, vmax)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
        cb.ax.tick_params(labelsize=7, length=2)
        cb.set_ticks([vmin, vmax])
        cb.ax.set_yticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
        if idx % 2 == 1:
            ax.set_yticklabels([])
        heat_axes.append(ax)

    heat_axes[0].text(-0.17, 1.16, "A", transform=heat_axes[0].transAxes, fontsize=11, fontweight="bold")
    fig.text(0.5, 0.985, FIG4A_CONFIG_LABEL, ha="center", va="top", fontsize=10, fontweight="bold")

    corr_branches = ["AAC", "DPC", "ESM2-150M", "ESM2-150M+handcrafted early fusion", "late fusion"]
    corr_labels = [BRANCH_LABELS.get(x, x) for x in corr_branches]
    matrix = pd.DataFrame(np.eye(len(corr_branches)), index=corr_branches, columns=corr_branches)
    internal_corr = corr_df[
        corr_df["Dataset_ID"].eq("internal_strict_nomixed_oof")
        & corr_df["Branch_A"].isin(corr_branches)
        & corr_df["Branch_B"].isin(corr_branches)
    ]
    for _, row in internal_corr.iterrows():
        a, b = row["Branch_A"], row["Branch_B"]
        val = row["Spearman_Rank_Correlation"]
        matrix.loc[a, b] = val
        matrix.loc[b, a] = val
    corr_ax = fig.add_subplot(corr_gs[0])
    corr_im = corr_ax.imshow(matrix.to_numpy(dtype=float), cmap="cividis", vmin=0, vmax=1)
    corr_ax.set_title("Internal OOF prediction-probability Spearman correlation", pad=3)
    corr_ax.set_xticks(np.arange(len(corr_branches)))
    corr_ax.set_xticklabels(corr_labels, rotation=35, ha="right")
    corr_ax.set_yticks(np.arange(len(corr_branches)))
    corr_ax.set_yticklabels(corr_labels)
    for i in range(len(corr_branches)):
        for j in range(len(corr_branches)):
            val = matrix.iloc[i, j]
            corr_ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="white" if val > 0.58 else "black")
    corr_ax.tick_params(length=0)
    for spine in corr_ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_subplot(corr_gs[1])
    cb = fig.colorbar(corr_im, cax=cax)
    cb.ax.tick_params(labelsize=8)
    corr_ax.text(-0.18, 1.13, "B", transform=corr_ax.transAxes, fontsize=11, fontweight="bold")

    roc_perm = perm_df[perm_df["Metric"].eq("ROC_AUC")].copy()
    roc_perm = roc_perm[roc_perm["Permuted_Group"].isin(["ESM2_150M", "AAC", "DPC"])]
    roc_perm["Facet"] = np.where(roc_perm["Permuted_Group"].eq("ESM2_150M"), "PLM feature group", "Handcrafted groups")
    roc_perm["External_Set"] = pd.Categorical(
        roc_perm["External_Set"],
        ["strict_novel40_N66", "seq2topt_noexact_N136", "public_topt_noexact_N984"],
        ordered=True,
    )
    marker_map = {"ESM2_150M": "o", "AAC": "s", "DPC": "^"}
    color_map = {"ESM2_150M": MODEL_COLORS["ESM2-150M"], "AAC": MODEL_COLORS["AAC"], "DPC": MODEL_COLORS["DPC"]}

    for ax_idx, facet in enumerate(["PLM feature group", "Handcrafted groups"]):
        ax = fig.add_subplot(perm_gs[ax_idx])
        sub = roc_perm[roc_perm["Facet"].eq(facet)].sort_values(["Permuted_Group", "External_Set"])
        groups = ["ESM2_150M"] if facet.startswith("PLM") else ["AAC", "DPC"]
        x = np.arange(3)
        offsets = [0.0] if len(groups) == 1 else [-0.11, 0.11]
        y_lows = []
        y_highs = []
        for group, offset in zip(groups, offsets):
            g = sub[sub["Permuted_Group"].eq(group)].sort_values("External_Set")
            y = g["Permutation_Effect_Degradation"].to_numpy(dtype=float)
            yerr = g["Permuted_Std"].to_numpy(dtype=float)
            y_lows.extend((y - yerr).tolist())
            y_highs.extend((y + yerr).tolist())
            ax.errorbar(
                x + offset,
                y,
                yerr=yerr,
                marker=marker_map[group],
                color=color_map[group],
                linestyle="-" if group == "ESM2_150M" else "--",
                linewidth=1.0,
                markersize=4.0,
                capsize=3,
                label=group.replace("_150M", "-150M"),
            )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(["N=66", "N=136", "N=984"])
        ax.set_title(facet)
        ax.set_ylabel("ROC-AUC degradation")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right")
        y_min = min(0.0, min(y_lows) if y_lows else 0.0)
        y_max = max(y_highs) if y_highs else 1.0
        margin = max((y_max - y_min) * 0.08, 0.002)
        ax.set_ylim(y_min - margin, y_max + margin)
        if ax_idx == 0:
            ax.text(-0.18, 1.13, "C", transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.text(0.5, 0.016, FIG4_FOOTNOTE, ha="center", va="bottom", fontsize=7, fontstyle="italic")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.93, bottom=0.08)

    outputs = save_all(fig, "fig4_ablation_redundancy_esm2_150m")
    plt.close(fig)

    qa_rows = [
        {
            "Figure": "Fig4",
            "Panel": "A",
            "Model_Filter": "ESM2-150M main branches only; all 650M branches excluded",
            "Branches": "; ".join(branch_filter),
            "Metrics": "; ".join(metric_order),
            "Color_Scale": "Per-metric normalization; Brier colormap reversed so lower values use the better-color direction",
        },
        {
            "Figure": "Fig4",
            "Panel": "B",
            "Model_Filter": "Internal OOF Spearman; no 650M branches",
            "Branches": "; ".join(corr_branches),
        },
        {
            "Figure": "Fig4",
            "Panel": "C",
            "Scale": "Faceted linear y-axis, no log axis",
            "Error_Bars": "Permuted_Std, N_Repeats=30",
            "YAxis": "Allows negative error-bar extent below zero; not clipped at y=0",
        },
    ]
    return {"outputs": outputs, "qa": qa_rows}


def make_alignment_table() -> Path:
    fig3 = pd.read_csv(FIG3_SOURCE)
    fig4 = pd.read_csv(FIG4_HEATMAP_SOURCE)
    fig3 = fig3[fig3["Embedding_Model"].eq("ESM2_150M")].copy()
    set_map = {
        "strict_novel40": "strict_novel40_N66",
        "seq2topt_noexact": "seq2topt_noexact_N136",
        "public_topt_noexact": "public_topt_noexact_N984",
    }
    metrics = [
        ("ROC_AUC", "ROC_AUC"),
        ("PR_AUC", "PR_AUC"),
        ("MCC", "MCC"),
        ("F1", "F1"),
        ("Balanced_Accuracy", "Balanced_Accuracy"),
        ("Brier_Score", "Brier"),
    ]
    rows = []
    for _, row in fig3.sort_values("External_Set").iterrows():
        fig3_set = row["External_Set"]
        fig4_set = set_map[fig3_set]
        for fig3_metric, fig4_metric in metrics:
            f4_row = fig4[
                fig4["Dataset_ID"].eq(fig4_set)
                & fig4["Branch"].eq("ESM2-150M")
                & fig4["Metric"].eq(fig4_metric)
            ]
            fig3_value = float(row[fig3_metric])
            fig4_value = float(f4_row["Value"].iloc[0])
            rows.append(
                {
                    "External_Set": fig3_set,
                    "Fig3_Dataset_ID": fig3_set,
                    "Fig4A_Dataset_ID": fig4_set,
                    "Metric": fig4_metric,
                    "Fig3_Value": fig3_value,
                    "Fig4A_Value": fig4_value,
                    "Difference_Fig4A_minus_Fig3": fig4_value - fig3_value,
                    "Same_Value_Tolerance_1e12": abs(fig4_value - fig3_value) <= 1e-12,
                    "Fig3_Source": "phase1_external_bootstrap_20260713/external_validation_main_table.csv",
                    "Fig4A_Source": "phase4_ablation_20260713/fig4_fusion_ablation_metric_long_table.csv",
                    "Interpretation": "different frozen analysis contexts: Phase1 external headline model versus Phase4 controlled ablation fixed-RF branch",
                }
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig3_vs_fig4a_esm2_150m_value_alignment.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main() -> None:
    set_ieee_style()
    fig3 = make_fig3()
    fig4 = make_fig4()
    alignment_path = make_alignment_table()

    qa = pd.DataFrame(fig3["qa"] + fig4["qa"])
    qa_path = OUT_DIR / "phase7_fig3_fig4_plot_qa.csv"
    qa.to_csv(qa_path, index=False)

    manifest_rows = []
    for figure, result in [("Fig3", fig3), ("Fig4", fig4)]:
        for path in result["outputs"]:
            manifest_rows.append({"Figure": figure, "Output_File": str(path), "Exists": path.exists(), "Bytes": path.stat().st_size if path.exists() else 0})
    manifest_rows.append({"Figure": "QA", "Output_File": str(qa_path), "Exists": qa_path.exists(), "Bytes": qa_path.stat().st_size if qa_path.exists() else 0})
    manifest_rows.append({"Figure": "Alignment", "Output_File": str(alignment_path), "Exists": alignment_path.exists(), "Bytes": alignment_path.stat().st_size if alignment_path.exists() else 0})
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "phase7_fig3_fig4_outputs_manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
