from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
OUT_DIR = ROOT / "figures" / "phase7_final"
SOURCE = ROOT / "results" / "analysis" / "phase2" / "screening_main_table.csv"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "bluish_green": "#009E73",
    "black": "#000000",
}

ESM2_150M_COLOR = OKABE_ITO["blue"]

REQUIRED_COLUMNS = [
    "Embedding_Model",
    "Decoy_Condition",
    "Ratio_Positive_to_Negative",
    "Ratio_Negatives_Per_Positive",
    "Top_Fraction",
    "Main_Text_Eligible",
    "EF_Mean",
    "EF_CI95_Lower",
    "EF_CI95_Upper",
    "EF_Max",
    "Precision_TopK_Mean",
    "Precision_TopK_CI95_Lower",
    "Precision_TopK_CI95_Upper",
    "Recall_TopK_Mean",
    "Recall_TopK_CI95_Lower",
    "Recall_TopK_CI95_Upper",
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
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def load_fig5_data() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required Fig5 screening columns: {missing}")

    sub = df[
        df["Embedding_Model"].eq("ESM2_150M")
        & df["Decoy_Condition"].eq("broad_lt60")
        & df["Top_Fraction"].isin([0.05, 0.10])
        & df["Main_Text_Eligible"].eq(True)
    ].copy()
    sub = sub.sort_values(["Ratio_Negatives_Per_Positive", "Top_Fraction"])
    if len(sub) != 6:
        raise ValueError(f"Expected 6 Fig5 main rows, got {len(sub)}")
    if sub["EF_CI95_Lower"].lt(0).any() or sub["Precision_TopK_CI95_Lower"].lt(0).any() or sub["Recall_TopK_CI95_Lower"].lt(0).any():
        raise ValueError("Detected negative CI lower bound in Fig5 screening data.")
    return sub


def errorbar_components(values: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    lower = values - lows
    upper = highs - values
    if np.any(lower < -1e-12) or np.any(upper < -1e-12):
        raise ValueError("Negative error-bar component detected.")
    return np.vstack([lower, upper])


def make_fig5() -> tuple[list[Path], Path]:
    sub = load_fig5_data()
    ratios = sorted(sub["Ratio_Negatives_Per_Positive"].unique())
    ratio_labels = [f"1:{int(r)}" for r in ratios]
    top_order = [0.05, 0.10]
    top_labels = {0.05: "Top-5%", 0.10: "Top-10%"}
    markers = {0.05: "o", 0.10: "s"}
    linestyles = {0.05: "-", 0.10: "--"}

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.75),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.32, 1.0, 1.0]},
    )

    metric_specs = [
        ("EF_Mean", "EF_CI95_Lower", "EF_CI95_Upper", "Enrichment factor", "A"),
        ("Precision_TopK_Mean", "Precision_TopK_CI95_Lower", "Precision_TopK_CI95_Upper", "Precision@Top-K", "B"),
        ("Recall_TopK_Mean", "Recall_TopK_CI95_Lower", "Recall_TopK_CI95_Upper", "Recall@Top-K", "C"),
    ]
    x = np.arange(len(ratios))

    for ax, (value_col, lo_col, hi_col, ylabel, panel_letter) in zip(axes, metric_specs):
        for top_frac in top_order:
            rows = sub[sub["Top_Fraction"].eq(top_frac)].set_index("Ratio_Negatives_Per_Positive").loc[ratios]
            y = rows[value_col].to_numpy(dtype=float)
            lo = rows[lo_col].to_numpy(dtype=float)
            hi = rows[hi_col].to_numpy(dtype=float)
            yerr = errorbar_components(y, lo, hi)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=markers[top_frac],
                linestyle=linestyles[top_frac],
                color=ESM2_150M_COLOR,
                markerfacecolor="white" if top_frac == 0.10 else ESM2_150M_COLOR,
                markeredgecolor=ESM2_150M_COLOR,
                linewidth=1.2,
                markersize=4.5,
                capsize=3,
                label=top_labels[top_frac],
            )

        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(ratio_labels)
        ax.set_xlabel("Spike-in pool ratio")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.text(-0.18, 1.08, panel_letter, transform=ax.transAxes, fontsize=11, fontweight="bold")

        if value_col == "EF_Mean":
            ef_max = sub.groupby("Ratio_Negatives_Per_Positive")["EF_Max"].first().loc[ratios].to_numpy(dtype=float)
            ef_ci_upper = float(sub["EF_CI95_Upper"].max())
            ef_ymax = 20.0 if ef_ci_upper > 18.0 else 18.0
            ax.set_ylim(0, ef_ymax)
            annotation_y = ef_ymax * 0.985
            n_ratios = len(x)
            for i, (xi, max_value) in enumerate(zip(x, ef_max)):
                # Anchor edge labels inside the axes so they cannot run into the
                # y tick labels (left) or past the right spine.
                if i == 0:
                    ha, dx = "left", 0.04
                elif i == n_ratios - 1:
                    ha, dx = "right", -0.04
                else:
                    ha, dx = "center", 0.0
                ax.plot(
                    xi,
                    annotation_y,
                    marker="v",
                    color=OKABE_ITO["black"],
                    markersize=3.8,
                    clip_on=False,
                    zorder=4,
                )
                ax.text(
                    xi + dx,
                    annotation_y - ef_ymax * 0.065,
                    f"EFmax={int(round(max_value))}",
                    ha=ha,
                    va="top",
                    fontsize=7,
                )
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.62, 0.85), borderaxespad=0.0)
        else:
            ax.set_ylim(0, 1.0)
            ax.legend(frameon=False, loc="upper right")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preview = OUT_DIR / "fig5_screening_utility_esm2_150m_broad_lt60_preview.png"
    pdf = OUT_DIR / "fig5_screening_utility_esm2_150m_broad_lt60.pdf"
    eps = OUT_DIR / "fig5_screening_utility_esm2_150m_broad_lt60.eps"
    caption = "The theoretical EF maximum equals 1/base rate and is annotated above each spike-in ratio."
    fig.text(0.5, -0.055, caption, ha="center", va="top", fontsize=7, fontstyle="italic")
    fig.savefig(pdf)
    fig.savefig(eps)
    fig.savefig(preview, dpi=600)
    plt.close(fig)

    qa = sub[
        [
            "Embedding_Model",
            "Decoy_Condition",
            "Ratio_Positive_to_Negative",
            "Ratio_Negatives_Per_Positive",
            "Top_Fraction",
            "EF_Mean",
            "EF_CI95_Lower",
            "EF_CI95_Upper",
            "EF_Max",
            "Precision_TopK_Mean",
            "Precision_TopK_CI95_Lower",
            "Precision_TopK_CI95_Upper",
            "Recall_TopK_Mean",
            "Recall_TopK_CI95_Lower",
            "Recall_TopK_CI95_Upper",
        ]
    ].copy()
    qa["EF_Max_Definition"] = "1 / Base_Positive_Rate = Pool_Size / Positives_In_Pool"
    qa_path = OUT_DIR / "fig5_screening_utility_qa.csv"
    qa.to_csv(qa_path, index=False)
    return [pdf, eps, preview], qa_path


def main() -> None:
    set_ieee_style()
    outputs, qa_path = make_fig5()
    manifest = pd.DataFrame(
        [{"Output_File": str(path), "Exists": path.exists(), "Bytes": path.stat().st_size if path.exists() else 0} for path in outputs + [qa_path]]
    )
    manifest.to_csv(OUT_DIR / "fig5_screening_utility_manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
