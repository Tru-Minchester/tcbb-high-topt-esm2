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

RELIABILITY_SOURCE = ROOT / "results" / "analysis" / "phase3" / "reliability_diagram_data.csv"
METRICS_SOURCE = ROOT / "results" / "analysis" / "phase3" / "calibration_metrics.csv"
EC_SOURCE = ROOT / "results" / "analysis" / "phase6" / "phase6_ec_stratified_metrics.csv"

CAPTION = (
    "Panel A: external calibration degrades on the small external sets (N=66, N=136), while the\n"
    "largest external set (N=984, ECE=0.044) remains close to calibrated; this pattern is consistent\n"
    "with internal-external distribution shift but is not by itself a causal test of shift. Post-hoc\n"
    "isotonic recalibration is unstable on the small sets due to sparse bins. Panel B: internal EC-stratified MCC\n"
    "(ESM2-150M); classes flagged as low-power exploratory (e.g. EC7) are shown with open markers and\n"
    "should not be over-interpreted."
)

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "bluish_green": "#009E73",
    "grey": "#6E6E6E",
    "black": "#000000",
}

METHOD_STYLES = {
    "uncalibrated": {"color": OKABE_ITO["blue"], "marker": "o", "linestyle": "-", "label": "Uncalibrated"},
    "isotonic": {"color": OKABE_ITO["orange"], "marker": "s", "linestyle": "--", "label": "Isotonic"},
}

DATASET_ORDER = ["strict_novel40", "seq2topt_noexact", "public_topt_noexact"]
DATASET_LABELS = {
    "strict_novel40": "N=66",
    "seq2topt_noexact": "N=136",
    "public_topt_noexact": "N=984",
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
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def load_calibration_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rel = pd.read_csv(RELIABILITY_SOURCE)
    metrics = pd.read_csv(METRICS_SOURCE)
    rel = rel[
        rel["Embedding_Model"].eq("ESM2_150M")
        & rel["Analysis_Role"].eq("primary")
        & rel["Binning_Label"].eq("equal_width_10")
        & rel["Calibration_Method"].isin(["uncalibrated", "isotonic"])
    ].copy()
    metrics = metrics[
        metrics["Embedding_Model"].eq("ESM2_150M")
        & metrics["Analysis_Role"].eq("primary")
        & metrics["Calibration_Method"].isin(["uncalibrated", "isotonic"])
    ].copy()
    if rel.empty or metrics.empty:
        raise ValueError("Missing Fig6 calibration rows after filtering.")
    rel["External_Set"] = pd.Categorical(rel["External_Set"], DATASET_ORDER, ordered=True)
    rel = rel.sort_values(["External_Set", "Calibration_Method", "Bin_Index"])
    return rel, metrics


def load_ec_data() -> pd.DataFrame:
    ec = pd.read_csv(EC_SOURCE)
    ec = ec[ec["Model_Branch"].eq("ESM2-150M") & ec["Analysis_Role"].eq("primary")].copy()
    if ec.empty:
        raise ValueError("Missing Fig6 EC rows after filtering.")
    ec = ec.sort_values("EC_Level1")
    return ec


def marker_sizes_from_bin_counts(n_bin: pd.Series, min_size: float = 14.0, max_size: float = 52.0) -> np.ndarray:
    values = n_bin.to_numpy(dtype=float)
    if len(values) == 0:
        return values
    vmax = max(values.max(), 1.0)
    return min_size + (max_size - min_size) * np.sqrt(values / vmax)


def plot_segmented_reliability_line(ax: plt.Axes, rows: pd.DataFrame, style: dict[str, str]) -> None:
    rows = rows.sort_values("Bin_Index")
    break_id = rows["Bin_Index"].diff().fillna(1).ne(1).cumsum()
    for _, segment in rows.groupby(break_id):
        ax.plot(
            segment["Mean_Predicted_Probability"],
            segment["Observed_Positive_Fraction"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=0.9,
            zorder=2,
        )
    ax.scatter(
        rows["Mean_Predicted_Probability"],
        rows["Observed_Positive_Fraction"],
        s=marker_sizes_from_bin_counts(rows["N_Bin"]),
        color=style["color"],
        marker=style["marker"],
        edgecolors=style["color"],
        linewidths=0.6,
        zorder=3,
    )


def make_fig6() -> tuple[list[Path], Path]:
    rel, metrics = load_calibration_data()
    ec = load_ec_data()

    fig = plt.figure(figsize=(7.16, 3.65), constrained_layout=False)
    cal_axes = [
        fig.add_axes([0.075, 0.31, 0.125, 0.50]),
        fig.add_axes([0.225, 0.31, 0.125, 0.50]),
        fig.add_axes([0.375, 0.31, 0.125, 0.50]),
    ]
    ax_ec = fig.add_axes([0.585, 0.31, 0.38, 0.50])
    fig.text(0.045, 0.94, "A", fontsize=11, fontweight="bold")
    fig.text(0.075, 0.94, "External calibration", fontsize=9, fontweight="bold")

    uncal = metrics[metrics["Calibration_Method"].eq("uncalibrated")].copy()
    empty_bin_rows = []
    for ax_idx, (ax_cal, external_set) in enumerate(zip(cal_axes, DATASET_ORDER)):
        ax_cal.plot([0, 1], [0, 1], color="#9E9E9E", linewidth=0.8, linestyle="-", label="Ideal y=x", zorder=1)
        for method, style in METHOD_STYLES.items():
            rows = rel[rel["External_Set"].eq(external_set) & rel["Calibration_Method"].eq(method)]
            if rows.empty:
                continue
            empty_bins = int(rows["N_Bin"].eq(0).sum())
            empty_bin_rows.append({"External_Set": external_set, "Calibration_Method": method, "Empty_Bins_Dropped": empty_bins})
            rows = rows[rows["N_Bin"].gt(0)].copy()
            plot_segmented_reliability_line(ax_cal, rows, style)

        metric_row = uncal[uncal["External_Set"].eq(external_set)].iloc[0]
        ax_cal.set_title(
            f"{DATASET_LABELS[external_set]}\nBrier={metric_row['Brier_Score']:.3f}\nECE={metric_row['ECE_equal_width_10']:.3f}",
            pad=3,
            fontsize=7.5,
        )
        ax_cal.set_xlabel("")
        ax_cal.set_xticks([0.0, 0.5, 1.0])
        if ax_idx == 0:
            ax_cal.set_ylabel("Observed positive fraction")
        else:
            ax_cal.set_yticklabels([])
        ax_cal.set_xlim(0, 1)
        ax_cal.set_ylim(0, 1)
        ax_cal.grid(color="#D9D9D9", linewidth=0.5)
        ax_cal.set_axisbelow(True)

    legend_handles = [
        plt.Line2D([0], [0], color="#9E9E9E", linewidth=0.8, label="Ideal y=x"),
        plt.Line2D([0], [0], color=METHOD_STYLES["uncalibrated"]["color"], marker="o", linestyle="-", linewidth=0.9, label="Uncalibrated"),
        plt.Line2D([0], [0], color=METHOD_STYLES["isotonic"]["color"], marker="s", linestyle="--", linewidth=0.9, label="Isotonic"),
    ]
    cal_axes[-1].legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=6.5)
    fig.text(0.287, 0.205, "Mean predicted probability", ha="center", va="top", fontsize=8)

    y_pos = np.arange(len(ec))
    main = ec["Power_Tier"].ne("low_power_exploratory")
    low = ec["Power_Tier"].eq("low_power_exploratory")
    ax_ec.scatter(
        ec.loc[main, "MCC"],
        y_pos[main],
        color=OKABE_ITO["blue"],
        edgecolor=OKABE_ITO["blue"],
        s=36,
        label="Main/reported",
        zorder=3,
    )
    ax_ec.scatter(
        ec.loc[low, "MCC"],
        y_pos[low],
        facecolors="white",
        edgecolors=OKABE_ITO["grey"],
        linewidths=1.2,
        s=44,
        label="Exploratory / low power",
        zorder=4,
    )
    ax_ec.axvline(0, color="#BDBDBD", linewidth=0.7)
    ax_ec.set_title("Internal EC-stratified performance")
    ax_ec.set_xlabel("MCC")
    ax_ec.set_xlim(0, max(1.0, float(ec["MCC"].max()) + 0.16))
    ax_ec.set_yticks(y_pos)
    ax_ec.set_yticklabels([f"EC{int(v)}" for v in ec["EC_Level1"]])
    ax_ec.invert_yaxis()
    ax_ec.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    ax_ec.set_axisbelow(True)
    ax_ec.text(-0.16, 1.08, "B", transform=ax_ec.transAxes, fontsize=11, fontweight="bold")
    ax_ec.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.82), borderaxespad=0.0)

    for yi, row in zip(y_pos, ec.itertuples(index=False)):
        label_color = OKABE_ITO["grey"] if row.Power_Tier == "low_power_exploratory" else OKABE_ITO["black"]
        ax_ec.text(
            row.MCC + 0.025,
            yi,
            f"{row.EC_Level1_Name}\nN={row.N}, pos={row.Positives}",
            ha="left",
            va="center",
            fontsize=6.5,
            color=label_color,
        )

    fig.text(0.5, 0.145, CAPTION, ha="center", va="top", fontsize=7, fontstyle="italic")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig6_calibration_ec_esm2_150m.pdf"
    eps = OUT_DIR / "fig6_calibration_ec_esm2_150m.eps"
    preview = OUT_DIR / "fig6_calibration_ec_esm2_150m_preview.png"
    fig.savefig(pdf)
    fig.savefig(eps)
    fig.savefig(preview, dpi=600)
    plt.close(fig)

    qa_rows = []
    qa_rows.append(
        {
            "Panel": "A",
            "Source": str(RELIABILITY_SOURCE),
            "Filter": "Embedding_Model=ESM2_150M; Analysis_Role=primary; Binning_Label=equal_width_10; Calibration_Method in uncalibrated,isotonic",
            "Error_Bars": "none",
            "Marker_Size": "scaled_by_N_Bin",
        }
    )
    for row in empty_bin_rows:
        qa_rows.append(
            {
                "Panel": "A_empty_bins",
                "Source": str(RELIABILITY_SOURCE),
                "Filter": f"External_Set={row['External_Set']}; Calibration_Method={row['Calibration_Method']}; N_Bin=0 dropped before plotting",
                "Error_Bars": "none",
                "Marker_Size": "scaled_by_N_Bin",
                "Low_Power_EC_Classes": "",
                "Empty_Bins_Dropped": row["Empty_Bins_Dropped"],
            }
        )
    qa_rows.append(
        {
            "Panel": "A_metrics",
            "Source": str(METRICS_SOURCE),
            "Filter": "Embedding_Model=ESM2_150M; Analysis_Role=primary; Calibration_Method=uncalibrated",
            "Error_Bars": "none",
        }
    )
    qa_rows.append(
        {
            "Panel": "B",
            "Source": str(EC_SOURCE),
            "Filter": "Model_Branch=ESM2-150M; Analysis_Role=primary",
            "Error_Bars": "none",
            "Low_Power_EC_Classes": "; ".join([f"EC{int(r.EC_Level1)} {r.EC_Level1_Name}" for r in ec[low].itertuples(index=False)]),
        }
    )
    qa_path = OUT_DIR / "fig6_calibration_ec_qa.csv"
    pd.DataFrame(qa_rows).to_csv(qa_path, index=False)
    return [pdf, eps, preview], qa_path


def main() -> None:
    set_ieee_style()
    outputs, qa_path = make_fig6()
    manifest = pd.DataFrame(
        [{"Output_File": str(path), "Exists": path.exists(), "Bytes": path.stat().st_size if path.exists() else 0} for path in outputs + [qa_path]]
    )
    manifest.to_csv(OUT_DIR / "fig6_calibration_ec_manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
