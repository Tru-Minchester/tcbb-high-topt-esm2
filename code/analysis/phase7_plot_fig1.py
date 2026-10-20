from __future__ import annotations

from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
OUT_DIR = ROOT / "figures" / "phase7_final"

OKABE_ITO = {
    "blue": "#0072B2",
    "sky_blue": "#56B4E9",
    "orange": "#E69F00",
    "bluish_green": "#009E73",
    "grey": "#BDBDBD",
    "dark_grey": "#5F6368",
    "black": "#000000",
}

INTERNAL = OKABE_ITO["blue"]
EXTERNAL = OKABE_ITO["bluish_green"]
HANDCRAFTED = OKABE_ITO["orange"]
ESM650 = OKABE_ITO["sky_blue"]
ANNOTATION = OKABE_ITO["dark_grey"]
DIVIDER = OKABE_ITO["grey"]


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def add_round_box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    edge_color: str,
    face_color: str = "#FFFFFF",
    linewidth: float = 1.3,
    linestyle: str = "-",
):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        linewidth=linewidth,
        edgecolor=edge_color,
        facecolor=face_color,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    return patch


def add_box_text(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str] | None = None,
    tag: str | None = None,
    title_size: float = 8.2,
    line_size: float = 8.0,
    tag_size: float = 8.0,
    pad_top: float = 0.020,
) -> None:
    lines = lines or []
    title_lines = title.split("\n")
    title_top = y + h - pad_top
    for index, title_line in enumerate(title_lines):
        ax.text(
            x + 0.012,
            title_top - index * 0.034,
            title_line,
            ha="left",
            va="top",
            fontsize=title_size,
            fontweight="bold",
        )
    line_y = title_top - len(title_lines) * 0.037 - 0.004
    for line in lines:
        ax.text(x + 0.012, line_y, line, ha="left", va="top", fontsize=line_size)
        line_y -= 0.035
    if tag:
        tag_lines = tag.split("\n")
        tag_top = y + (0.050 if len(tag_lines) > 1 else 0.040)
        for index, tag_line in enumerate(tag_lines):
            ax.text(
                x + 0.012,
                tag_top - index * 0.026,
                tag_line,
                ha="left",
                va="top",
                fontsize=tag_size,
                color=ANNOTATION,
            )


def add_stage_title(ax, x: float, y: float, text: str) -> None:
    ax.text(x, y, text, ha="left", va="center", fontsize=8.6, fontweight="bold")


def arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str, dashed: bool = False) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 1.05,
            "linestyle": "--" if dashed else "-",
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def elbow_arrow(ax, points: list[tuple[float, float]], color: str, dashed: bool = False) -> None:
    if len(points) < 2:
        raise ValueError("elbow_arrow requires at least two points.")
    line_style = "--" if dashed else "-"
    if len(points) > 2:
        xs, ys = zip(*points[:-1])
        ax.plot(xs, ys, color=color, linewidth=1.05, linestyle=line_style)
    ax.annotate(
        "",
        xy=points[-1],
        xytext=points[-2],
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 1.05,
            "linestyle": line_style,
            "shrinkA": 0,
            "shrinkB": 2,
        },
    )


def make_fig1() -> list[Path]:
    set_ieee_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.16, 5.20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for x in [0.300, 0.535, 0.775]:
        ax.plot([x, x], [0.118, 0.912], color=DIVIDER, linewidth=0.8)

    add_stage_title(ax, 0.028, 0.958, "1. Data sources")
    add_stage_title(ax, 0.320, 0.958, "2. Feature extraction")
    add_stage_title(ax, 0.635, 0.958, "3. Modeling")
    add_stage_title(ax, 0.800, 0.958, "4. Evaluation")

    internal_xy = (0.035, 0.718)
    internal_wh = (0.248, 0.187)
    add_round_box(ax, internal_xy, internal_wh, INTERNAL)
    add_box_text(
        ax,
        *internal_xy,
        *internal_wh,
        "Internal benchmark",
        ["N=5,263 | pos=672,", "neg=4,591"],
        "CD-HIT 40%; mixed-label\nclusters removed",
        title_size=8.1,
        line_size=8.0,
        tag_size=7.8,
    )

    ax.text(0.040, 0.694, "evaluation only \u2192", ha="left", va="center", fontsize=8.0, color=EXTERNAL)
    external_items = [
        ("Strict novel40", "N=66 | pos=26,\nneg=40", "Most stringent\n(low-homology)", 0.492),
        ("Seq2Topt no-exact", "N=136 | pos=33,\nneg=103", "no-exact-overlap", 0.296),
        ("Public Topt no-exact", "N=984 | pos=313,\nneg=671", "no-exact-overlap", 0.100),
    ]
    for title, counts, tag, y in external_items:
        x, w, h = 0.035, 0.248, 0.176
        add_round_box(ax, (x, y), (w, h), EXTERNAL)
        add_box_text(ax, x, y, w, h, title, counts.split("\n"), tag,
                     title_size=8.0, line_size=8.0, tag_size=7.8)

    feature_boxes = [
        ("AAC", "20-dim", HANDCRAFTED, "-", None, 0.765),
        ("DPC", "400-dim", HANDCRAFTED, "-", None, 0.543),
        ("ESM2-150M", "640-dim", INTERNAL, "-", None, 0.321),
        ("ESM2-650M", "1280-dim", ESM650, "--", "sensitivity", 0.100),
    ]
    FX, FW, FH = 0.320, 0.195, 0.140
    for name, dim, color, linestyle, tag, y in feature_boxes:
        add_round_box(ax, (FX, y), (FW, FH), color, linewidth=1.3, linestyle=linestyle)
        ax.text(FX + 0.014, y + FH - 0.026, name, ha="left", va="top",
                fontsize=8.2, fontweight="bold")
        ax.text(FX + 0.014, y + 0.016, dim, ha="left", va="bottom",
                fontsize=8.0, color=ANNOTATION)
        if tag:
            ax.text(FX + FW - 0.012, y + 0.050, tag, ha="right", va="bottom",
                    fontsize=7.8, color=ANNOTATION, style="italic")

    data_bus_x = 0.291
    feature_mids = [y + FH / 2 for _, _, _, _, _, y in feature_boxes]
    ax.plot([data_bus_x, data_bus_x], [feature_mids[-1], feature_mids[0]], color=DIVIDER, linewidth=0.9)
    elbow_arrow(ax, [(internal_xy[0] + internal_wh[0], internal_xy[1] + internal_wh[1] / 2), (data_bus_x, internal_xy[1] + internal_wh[1] / 2)], INTERNAL)
    elbow_arrow(ax, [(0.283, 0.384), (data_bus_x, 0.384)], EXTERNAL, dashed=True)
    for (_, _, color, linestyle, _, _), mid_y in zip(feature_boxes, feature_mids):
        elbow_arrow(ax, [(data_bus_x, mid_y), (FX, mid_y)], color, dashed=(linestyle == "--"))

    MX, MW = 0.565, 0.175
    model_xy = (MX, 0.620)
    model_wh = (MW, 0.285)
    add_round_box(ax, model_xy, model_wh, INTERNAL)
    add_box_text(ax, *model_xy, *model_wh,
                 "Imputer \u2192 Scaler\n\u2192 SMOTE \u2192 RF",
                 ["5-fold OOF | fixed", "hyperparameters"],
                 None, title_size=8.0, line_size=8.0, pad_top=0.075)

    fusion_xy = (MX, 0.100)
    fusion_wh = (MW, 0.285)
    add_round_box(ax, fusion_xy, fusion_wh, INTERNAL)
    add_box_text(ax, *fusion_xy, *fusion_wh,
                 "Late fusion\n(stacking)",
                 ["OOF probabilities", "as meta-features"],
                 None, title_size=8.0, line_size=8.0, pad_top=0.075)

    model_bus_x = 0.545
    model_mid = model_xy[1] + model_wh[1] / 2
    fusion_mid = fusion_xy[1] + fusion_wh[1] / 2
    ax.plot([model_bus_x, model_bus_x], [fusion_mid, model_mid], color=DIVIDER, linewidth=0.9)
    elbow_arrow(ax, [(FX + FW, feature_mids[2]), (model_bus_x, feature_mids[2]), (model_bus_x, model_mid), (MX, model_mid)], INTERNAL)
    elbow_arrow(ax, [(FX + FW, feature_mids[3]), (model_bus_x, feature_mids[3]), (model_bus_x, fusion_mid), (MX, fusion_mid)], INTERNAL)

    EX, EW = 0.795, 0.190
    EH = 0.205
    eval_items = [
        ("External validation", "Bootstrap 95% CI\nROC-AUC / PR-AUC\nMCC / Brier", 0.700),
        ("Screening utility", "Enrichment factor\nPrecision & Recall\n@Top-K", 0.400),
        ("Calibration & EC\nstratification", "Reliability curves\nEC-class MCC", 0.100),
    ]
    eval_mids = []
    for title, subtitle, y in eval_items:
        add_round_box(ax, (EX, y), (EW, EH), EXTERNAL)
        add_box_text(ax, EX, y, EW, EH, title, subtitle.split("\n"),
                     None, title_size=8.0, line_size=7.5)
        eval_mids.append(y + EH / 2)

    eval_bus_x = 0.770
    ax.plot([eval_bus_x, eval_bus_x], [eval_mids[-1], eval_mids[0]], color=DIVIDER, linewidth=0.9)
    mr = MX + MW
    elbow_arrow(ax, [(mr, model_mid), (eval_bus_x, model_mid), (eval_bus_x, eval_mids[0]), (EX, eval_mids[0])], EXTERNAL)
    elbow_arrow(ax, [(mr, fusion_mid), (eval_bus_x, fusion_mid), (eval_bus_x, eval_mids[1]), (EX, eval_mids[1])], EXTERNAL)
    elbow_arrow(ax, [(mr, fusion_mid), (eval_bus_x, fusion_mid), (eval_bus_x, eval_mids[2]), (EX, eval_mids[2])], EXTERNAL)
    ax.text(
        0.6525,
        0.5025,
        "evaluation only\n(labels never used\nfor training)",
        ha="center",
        va="center",
        fontsize=8.0,
        color=EXTERNAL,
    )

    ax.text(
        0.028, 0.058,
        "Handcrafted descriptors and frozen PLM embeddings feed the same controlled RF framework; "
        "ESM2-650M is a model-scale sensitivity branch.",
        ha="left", va="center", fontsize=7.1, color=ANNOTATION,
    )

    outputs = [
        OUT_DIR / "fig1_system_architecture.pdf",
        OUT_DIR / "fig1_system_architecture.eps",
        OUT_DIR / "fig1_system_architecture_preview.png",
    ]
    fig.savefig(outputs[0])
    fig.savefig(outputs[1])
    fig.savefig(outputs[2], dpi=600)
    plt.close(fig)
    return outputs


def main() -> None:
    outputs = make_fig1()
    manifest = pd.DataFrame(
        {
            "Output_File": [str(path) for path in outputs],
            "Exists": [path.exists() for path in outputs],
            "Bytes": [path.stat().st_size if path.exists() else 0 for path in outputs],
        }
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
