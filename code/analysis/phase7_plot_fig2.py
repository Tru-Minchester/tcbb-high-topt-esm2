from __future__ import annotations

from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
OUT_DIR = ROOT / "figures" / "phase7_final"

INTERNAL_SOURCE = ROOT / "results" / "analysis" / "phase0" / "data_inventory.csv"
EXTERNAL_SOURCE = ROOT / "results" / "analysis" / "phase1" / "external_validation_main_table.csv"

OKABE_ITO = {
    "blue": "#0072B2",
    "sky_blue": "#56B4E9",
    "orange": "#E69F00",
    "bluish_green": "#009E73",
    "grey": "#BDBDBD",
    "dark_grey": "#5F6368",
    "black": "#000000",
}

POS_COLOR = OKABE_ITO["blue"]
NEG_COLOR = "#D0D0D0"
INTERNAL_EDGE = OKABE_ITO["blue"]
EXTERNAL_EDGE = OKABE_ITO["bluish_green"]

EXPECTED = {
    "internal_strict_nomixed": {"N": 5263, "Positives": 672, "Negatives": 4591},
    "strict_novel40": {"N": 66, "Positives": 26, "Negatives": 40},
    "seq2topt_noexact": {"N": 136, "Positives": 33, "Negatives": 103},
    "public_topt_noexact": {"N": 984, "Positives": 313, "Negatives": 671},
}


def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def load_counts() -> list[dict[str, object]]:
    internal = pd.read_csv(INTERNAL_SOURCE)
    external = pd.read_csv(EXTERNAL_SOURCE)

    row = internal[internal["file_name"].eq("13_cdhit40_strict_nomixed_dataset.csv")]
    if len(row) != 1:
        raise ValueError("Could not find internal strict no-mixed row in data inventory.")
    row = row.iloc[0]
    internal_item = {
        "id": "internal_strict_nomixed",
        "name": "Internal strict no-mixed benchmark",
        "short_name": "Internal benchmark",
        "N": int(row["rows"]),
        "Positives": int(row["positive_count"]),
        "Negatives": int(row["negative_count"]),
        "tag": "CD-HIT 40%; mixed-label\nclusters removed",
        "zone": "internal",
    }

    external = external[external["Embedding_Model"].eq("ESM2_150M")].copy()
    external_items = []
    external_names = {
        "strict_novel40": ("Strict Seq2Topt novel40 external", "Most stringent low-homology panel"),
        "seq2topt_noexact": ("Seq2Topt no-exact-overlap external", "no-exact-overlap"),
        "public_topt_noexact": ("Public Topt no-exact-overlap external", "no-exact-overlap"),
    }
    for external_id, (display_name, tag) in external_names.items():
        erow = external[external["External_Set"].eq(external_id)]
        if len(erow) != 1:
            raise ValueError(f"Could not find one ESM2_150M row for {external_id}.")
        erow = erow.iloc[0]
        external_items.append(
            {
                "id": external_id,
                "name": display_name,
                "short_name": display_name.replace(" external", ""),
                "N": int(erow["N"]),
                "Positives": int(erow["Positives"]),
                "Negatives": int(erow["Negatives"]),
                "tag": tag,
                "zone": "external",
            }
        )

    items = [internal_item] + external_items
    for item in items:
        expected = EXPECTED[item["id"]]
        actual = {key: int(item[key]) for key in ["N", "Positives", "Negatives"]}
        if actual != expected:
            raise ValueError(f"Count mismatch for {item['id']}: actual {actual}, expected {expected}")
        item["Positive_Rate"] = item["Positives"] / item["N"]
    return items


def add_round_box(ax, xy, wh, edge_color, face_color="#FFFFFF", linewidth=1.3):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=linewidth,
        edgecolor=edge_color,
        facecolor=face_color,
    )
    ax.add_patch(patch)
    return patch


def draw_dataset_box(ax, item: dict[str, object], x: float, y: float, w: float, h: float, edge_color: str) -> None:
    add_round_box(ax, (x, y), (w, h), edge_color=edge_color)
    pad_x = 0.025
    title_y = y + h - 0.042
    count_y = y + h - 0.085
    rate_y  = y + h - 0.168
    bar_y   = y + h - 0.155
    tag_y   = y + 0.022
    ax.text(x + pad_x, title_y, str(item["short_name"]), ha="left", va="top", fontsize=8, fontweight="bold")
    ax.text(
        x + pad_x,
        count_y,
        f"N={item['N']:,} | pos={item['Positives']:,}, neg={item['Negatives']:,}",
        ha="left",
        va="top",
        fontsize=7.5,
    )
    ax.text(
        x + pad_x,
        rate_y,
        f"positive rate={item['Positive_Rate'] * 100:.1f}%",
        ha="left",
        va="top",
        fontsize=7.5,
    )

    bar_x = x + pad_x
    bar_w = w - 0.05
    bar_h = 0.024
    pos_w = bar_w * float(item["Positive_Rate"])
    ax.add_patch(Rectangle((bar_x, bar_y), pos_w, bar_h, facecolor=POS_COLOR, edgecolor="none"))
    ax.add_patch(Rectangle((bar_x + pos_w, bar_y), bar_w - pos_w, bar_h, facecolor=NEG_COLOR, edgecolor="none"))
    ax.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h, facecolor="none", edgecolor=OKABE_ITO["dark_grey"], linewidth=0.45))

    ax.text(x + pad_x, tag_y, str(item["tag"]), ha="left", va="bottom", fontsize=6.5,
            color=OKABE_ITO["dark_grey"])


def make_fig2() -> tuple[list[Path], Path]:
    items = load_counts()
    by_id = {item["id"]: item for item in items}

    fig, ax = plt.subplots(figsize=(7.16, 3.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.945, "Internal benchmark", fontsize=10, fontweight="bold", ha="left", va="center")
    ax.text(0.56, 0.945, "External validation cohorts", fontsize=10, fontweight="bold", ha="left", va="center")
    ax.plot([0.50, 0.50], [0.13, 0.91], color="#BDBDBD", linewidth=0.9)
    ax.text(0.56, 0.920, "Independently curated external cohorts (not derived from the internal set)", ha="left", va="top", fontsize=7.1, color=OKABE_ITO["dark_grey"])

    draw_dataset_box(ax, by_id["internal_strict_nomixed"], 0.055, 0.54, 0.37, 0.32, INTERNAL_EDGE)
    ax.annotate(
        "",
        xy=(0.24, 0.43),
        xytext=(0.24, 0.54),
        arrowprops={"arrowstyle": "-|>", "color": INTERNAL_EDGE, "linewidth": 1.0},
    )
    add_round_box(ax, (0.10, 0.25), (0.28, 0.15), edge_color=INTERNAL_EDGE, face_color="#F8FBFD", linewidth=1.0)
    ax.text(0.24, 0.345, "Frozen internal CV", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(0.24, 0.295, "5-fold OOF evaluation\nRF/SMOTE model selection", ha="center", va="center", fontsize=7, color=OKABE_ITO["dark_grey"])

    external_positions = {
        "strict_novel40": (0.58, 0.620),
        "seq2topt_noexact": (0.58, 0.350),
        "public_topt_noexact": (0.58, 0.080),
    }
    box_w = 0.36
    box_h = 0.255
    for external_id, (x, y) in external_positions.items():
        draw_dataset_box(ax, by_id[external_id], x, y, box_w, box_h, EXTERNAL_EDGE)


    ax.annotate(
        "",
        xy=(0.575, 0.477),
        xytext=(0.515, 0.477),
        arrowprops={"arrowstyle": "-|>", "color": EXTERNAL_EDGE, "linewidth": 1.0},
    )
    ax.text(0.513, 0.477, "evaluation only", ha="right", va="center", fontsize=6.8, color=EXTERNAL_EDGE)

    legend_y = 0.045
    ax.add_patch(Rectangle((0.055, legend_y), 0.024, 0.018, facecolor=POS_COLOR, edgecolor="none"))
    ax.text(0.085, legend_y + 0.009, "High-Topt positive", va="center", fontsize=7)
    ax.add_patch(Rectangle((0.205, legend_y), 0.024, 0.018, facecolor=NEG_COLOR, edgecolor="none"))
    ax.text(0.235, legend_y + 0.009, "Negative / lower-Topt", va="center", fontsize=7)
    ax.text(
        0.55,
        0.020,
        "N=136 and N=984 are complementary larger no-exact panels, NOT replacements for N=66.\nBox area does not encode N; stacked bars show within-cohort class balance.",
        va="center",
        fontsize=6.8,
        color=OKABE_ITO["dark_grey"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig2_data_construction_validation_cohorts.pdf"
    eps = OUT_DIR / "fig2_data_construction_validation_cohorts.eps"
    preview = OUT_DIR / "fig2_data_construction_validation_cohorts_preview.png"
    fig.savefig(pdf)
    fig.savefig(eps)
    fig.savefig(preview, dpi=600)
    plt.close(fig)

    qa = pd.DataFrame(items)
    qa["Source"] = qa["id"].map(
        {
            "internal_strict_nomixed": str(INTERNAL_SOURCE),
            "strict_novel40": str(EXTERNAL_SOURCE),
            "seq2topt_noexact": str(EXTERNAL_SOURCE),
            "public_topt_noexact": str(EXTERNAL_SOURCE),
        }
    )
    qa_path = OUT_DIR / "fig2_data_construction_validation_cohorts_qa.csv"
    qa.to_csv(qa_path, index=False)
    return [pdf, eps, preview], qa_path


def main() -> None:
    set_ieee_style()
    outputs, qa_path = make_fig2()
    manifest = pd.DataFrame(
        [{"Output_File": str(path), "Exists": path.exists(), "Bytes": path.stat().st_size if path.exists() else 0} for path in outputs + [qa_path]]
    )
    manifest.to_csv(OUT_DIR / "fig2_data_construction_validation_cohorts_manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
