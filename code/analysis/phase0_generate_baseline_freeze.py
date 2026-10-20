from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
RELEASE = ROOT
SUBMISSION = ROOT / "docs"
OUT = ROOT / "results" / "analysis" / "phase0"


DATA_DIRS = [
    RELEASE / "data_processed",
    RELEASE / "features",
]

RESULT_DIR = RELEASE / "results"
FIGURE_DIRS = [
    RELEASE / "figures",
    ROOT,
]
WORD_DIRS = [
    ROOT,
    SUBMISSION,
    RELEASE / "docs",
]


KNOWN_ROLES = {
    "13_cdhit40_strict_nomixed_dataset.csv": "primary internal strict no-mixed benchmark",
    "14_strict_nomixed_clean_dataset.csv": "primary internal strict no-mixed clean sequence table",
    "15_strict_nomixed_esm2_features.csv": "primary ESM-2 150M feature matrix",
    "23_external_final_dataset.csv": "strict Seq2Topt-derived novel40 external set",
    "24_external_esm2_features.csv": "strict novel40 ESM-2 150M feature matrix",
    "27_strict_nomixed_esm2_650m_features.csv": "primary ESM-2 650M feature matrix",
    "28_external_esm2_650m_features.csv": "strict novel40 ESM-2 650M feature matrix",
    "32_external_noexact_esm2_150m_features.csv": "Seq2Topt no-exact external ESM-2 150M feature matrix",
    "32_external_noexact_esm2_650m_features.csv": "Seq2Topt no-exact external ESM-2 650M feature matrix",
    "33_public_topt_candidate_noexact.csv": "public Topt no-exact external candidate table",
    "34_public_topt_noexact_esm2_150m_features.csv": "public Topt no-exact ESM-2 150M feature matrix",
    "34_public_topt_noexact_esm2_650m_features.csv": "public Topt no-exact ESM-2 650M feature matrix",
}


FIGURE_NOTES = {
    "fig5_real_external_novel40_150m.png": (
        "real prediction-derived ROC/PR figure",
        "Generated from held-out strict novel40 per-sample predictions for ESM-2 150M; current PLOS Figure 5 basis.",
        "usable as historical/PLOS baseline; TCBB Fig. 5 should be redrawn to include all external sets",
    ),
    "fig5_real_external_novel40_650m.png": (
        "real prediction-derived ROC/PR figure",
        "Generated from held-out strict novel40 per-sample predictions for ESM-2 650M; alternative sensitivity figure.",
        "usable as sensitivity source; not current PLOS primary Figure 5",
    ),
    "fig5_real_external_novel40.png": (
        "real prediction-derived ROC/PR figure",
        "Earlier real Figure 5 export based on strict novel40 predictions.",
        "superseded by explicit 150M/650M exports",
    ),
}


KEY_RESULTS = [
    "16_strict_nomixed_feature_set_summary.csv",
    "25_external_test_results.csv",
    "26_esm2_topt_regression_summary.csv",
    "29_model_scale_internal_summary.csv",
    "29_model_scale_external_summary.csv",
    "30_external_bootstrap_ci.csv",
    "31_mixed_label_sensitivity_summary.csv",
    "32_external_noexact_model_scale_summary.csv",
    "34_public_topt_noexact_model_scale_summary.csv",
    "35_expanded_external_threshold_sensitivity.csv",
    "36_classifier_sensitivity_internal_summary.csv",
    "36_classifier_sensitivity_external_summary.csv",
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def detect_category(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    suffix = path.suffix.lower()
    if "data_processed" in parts:
        return "processed_data"
    if "features" in parts:
        return "feature_matrix"
    if "results" in parts:
        return "result_table"
    if "figures" in parts or suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return "figure"
    if suffix == ".docx":
        return "manuscript_or_docx"
    if suffix == ".py":
        return "code"
    if suffix in {".md", ".txt"}:
        return "documentation"
    return "other"


def sniff_csv(path: Path, max_rows: int = 200000) -> dict:
    info = {
        "rows": "",
        "columns": "",
        "sequence_column": "",
        "label_column": "",
        "temperature_column": "",
        "positive_count": "",
        "negative_count": "",
        "feature_dimension_estimate": "",
        "notes": "",
    }
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                info["notes"] = "no header detected"
                return info
            fields = reader.fieldnames
            info["columns"] = len(fields)
            lower = {c.lower(): c for c in fields}
            seq_col = next((lower[k] for k in lower if k in {"sequence", "seq", "protein_sequence"} or "sequence" in k), "")
            label_col = next((lower[k] for k in lower if k in {"label", "target", "is_high_topt", "high_topt", "class"} or "label" in k), "")
            temp_col = next((lower[k] for k in lower if "topt" in k or "temperature" in k or k in {"temp", "optimal_temperature"}), "")
            info["sequence_column"] = seq_col
            info["label_column"] = label_col
            info["temperature_column"] = temp_col
            label_counter: Counter[str] = Counter()
            n = 0
            for row in reader:
                n += 1
                if label_col:
                    val = str(row.get(label_col, "")).strip()
                    label_counter[val] += 1
                if n >= max_rows:
                    info["notes"] = f"row count truncated at {max_rows}"
                    break
            info["rows"] = n
            if label_counter:
                pos = sum(v for k, v in label_counter.items() if k in {"1", "1.0", "true", "True", "positive", "Positive", "pos"})
                neg = sum(v for k, v in label_counter.items() if k in {"0", "0.0", "false", "False", "negative", "Negative", "neg"})
                info["positive_count"] = pos if pos else ""
                info["negative_count"] = neg if neg else ""
            if "feature" in detect_category(path) or "features" in path.name.lower():
                prefixed = [
                    c for c in fields
                    if re.match(r"^(ESM|AAC|DPC|CTD|AAINDEX|PHYS|PC)_?\d+", c, re.I)
                    or re.match(r"^ESM_\d+$", c, re.I)
                ]
                non_feature = {
                    "id", "accession", "uniprot_accession", "sequence_hash", "sequence", "seq",
                    "label", "binary_label", "target", "target_temperature", "topt",
                    "temperature", "optimal_temperature", "is_high_topt", "high_topt",
                    "source", "dataset", "sequence_length", "was_truncated"
                }
                feature_cols = prefixed or [
                    c for c in fields
                    if c.lower() not in non_feature and not c.lower().endswith("_id")
                ]
                info["feature_dimension_estimate"] = len(feature_cols)
    except UnicodeDecodeError:
        info["notes"] = "not UTF-8 text or binary-like CSV"
    except Exception as exc:
        info["notes"] = f"inspect error: {type(exc).__name__}: {exc}"
    return info


def csv_head(path: Path, n: int = 20) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return [row for _, row in zip(range(n), reader)]
    except Exception:
        return []


def read_small_text(path: Path, limit: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        return text[:limit].strip()
    except Exception:
        return ""


def docx_info(path: Path) -> dict:
    info = {
        "paragraphs": "",
        "tables": "",
        "images": "",
        "highlight_elements": "",
        "embedded_image_md5": "",
        "notes": "",
    }
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            media = [n for n in names if n.startswith("word/media/")]
            info["images"] = len(media)
            info["embedded_image_md5"] = "; ".join(
                f"{Path(n).name}:{hashlib.md5(z.read(n)).hexdigest()}" for n in media
            )
            xml = z.read("word/document.xml").decode("utf-8", errors="replace")
            info["paragraphs"] = xml.count("<w:p")
            info["tables"] = xml.count("<w:tbl")
            info["highlight_elements"] = xml.count("<w:highlight")
            dirty = [s for s in ["ЎЭ", "Ўг", "ЎА"] if s in xml]
            if dirty:
                info["notes"] = "possible mojibake symbols in document XML: " + ",".join(dirty)
    except Exception as exc:
        info["notes"] = f"docx inspect error: {type(exc).__name__}: {exc}"
    return info


def collect_file_inventory() -> list[dict]:
    roots = [
        RELEASE,
        SUBMISSION,
    ]
    extra_files = [p for p in ROOT.iterdir() if p.is_file()]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend([p for p in root.rglob("*") if p.is_file()])
    files.extend(extra_files)
    seen: set[Path] = set()
    rows = []
    for path in sorted(files):
        if path in seen:
            continue
        seen.add(path)
        stat = path.stat()
        rows.append(
            {
                "relative_path": rel(path),
                "file_name": path.name,
                "category": detect_category(path),
                "role": KNOWN_ROLES.get(path.name, ""),
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "last_write_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "md5": md5(path),
            }
        )
    return rows


def collect_data_inventory(file_rows: list[dict]) -> list[dict]:
    rows = []
    for fr in file_rows:
        if fr["category"] not in {"processed_data", "feature_matrix"}:
            continue
        path = ROOT / fr["relative_path"]
        row = {k: fr[k] for k in ["relative_path", "file_name", "category", "role", "size_bytes", "last_write_time", "md5"]}
        if path.suffix.lower() == ".csv":
            row.update(sniff_csv(path))
        else:
            row.update({
                "rows": "",
                "columns": "",
                "sequence_column": "",
                "label_column": "",
                "temperature_column": "",
                "positive_count": "",
                "negative_count": "",
                "feature_dimension_estimate": "",
                "notes": "non-CSV data artifact",
            })
        rows.append(row)
    return rows


def collect_results_inventory(file_rows: list[dict]) -> list[dict]:
    rows = []
    for fr in file_rows:
        if fr["category"] != "result_table":
            continue
        path = ROOT / fr["relative_path"]
        row = {k: fr[k] for k in ["relative_path", "file_name", "category", "role", "size_bytes", "last_write_time", "md5"]}
        if path.suffix.lower() == ".csv":
            info = sniff_csv(path)
            rows_head = csv_head(path, 2)
            fields = []
            if rows_head:
                fields = list(rows_head[0].keys())
            metric_cols = [c for c in fields if re.search(r"auc|mcc|f1|precision|recall|brier|balanced|mae|rmse|r2|pearson|spearman|ci", c, re.I)]
            row.update(
                {
                    "rows": info["rows"],
                    "columns": info["columns"],
                    "metric_columns_detected": "; ".join(metric_cols),
                    "first_rows_json": json.dumps(rows_head, ensure_ascii=False)[:3000],
                    "notes": info["notes"],
                }
            )
        else:
            row.update(
                {
                    "rows": "",
                    "columns": "",
                    "metric_columns_detected": "",
                    "first_rows_json": read_small_text(path, 1000),
                    "notes": "non-CSV result artifact",
                }
            )
        rows.append(row)
    return rows


def collect_figure_inventory(file_rows: list[dict]) -> list[dict]:
    rows = []
    for fr in file_rows:
        path = ROOT / fr["relative_path"]
        if fr["category"] == "figure":
            origin, note, action = FIGURE_NOTES.get(
                fr["file_name"],
                (
                    "unknown or legacy visual asset",
                    "Origin not certified in Phase 0 scan.",
                    "review before using in TCBB figures",
                ),
            )
            rows.append(
                {
                    "relative_path": fr["relative_path"],
                    "file_name": fr["file_name"],
                    "asset_type": "standalone image",
                    "origin_status": origin,
                    "current_use": note,
                    "tcbb_action": action,
                    "size_bytes": fr["size_bytes"],
                    "last_write_time": fr["last_write_time"],
                    "md5": fr["md5"],
                    "embedded_or_standalone": "standalone",
                }
            )
        elif fr["extension"] == ".docx":
            info = docx_info(path)
            rows.append(
                {
                    "relative_path": fr["relative_path"],
                    "file_name": fr["file_name"],
                    "asset_type": "docx manuscript",
                    "origin_status": "contains embedded figures",
                    "current_use": f"images={info['images']}; highlights={info['highlight_elements']}; paragraphs={info['paragraphs']}; tables={info['tables']}",
                    "tcbb_action": "use only as manuscript baseline; TCBB figures should be regenerated as standalone publication figures",
                    "size_bytes": fr["size_bytes"],
                    "last_write_time": fr["last_write_time"],
                    "md5": fr["md5"],
                    "embedded_or_standalone": "embedded in docx",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def metric_value(rows: list[dict], *needles: str) -> str:
    for row in rows:
        hay = " ".join(str(v) for v in row.values()).lower()
        if all(n.lower() in hay for n in needles):
            return json.dumps(row, ensure_ascii=False)
    return ""


def load_result(name: str) -> list[dict]:
    path = RESULT_DIR / name
    return csv_head(path, 200) if path.exists() else []


def make_baseline_summary(file_rows, data_rows, result_rows, figure_rows) -> str:
    summaries = {
        name: load_result(name) for name in KEY_RESULTS
    }
    data_counts = {
        "primary_internal": next((r for r in data_rows if r["file_name"] == "13_cdhit40_strict_nomixed_dataset.csv"), {}),
        "strict_external": next((r for r in data_rows if r["file_name"] == "23_external_final_dataset.csv"), {}),
        "public_external": next((r for r in data_rows if r["file_name"] == "33_public_topt_candidate_noexact.csv"), {}),
    }
    docx_rows = [r for r in file_rows if r["extension"] == ".docx"]
    current_submission = SUBMISSION / "Manuscript.docx"
    highlighted_submission = SUBMISSION / "Revised Manuscript with Highlighted Changes.docx"
    response_doc = SUBMISSION / "Response to Reviewers.docx"

    lines = [
        "# Phase 0 Baseline Freeze",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Purpose",
        "",
        "This folder freezes the current PLOS revision baseline before TCBB-oriented expansion. It records the exact data, result, figure, and manuscript artifacts currently available so later bootstrap, screening, calibration, fusion, interpretability, EC-stratified, and figure-redrawing work can be traced without mixing old and new outputs.",
        "",
        "## Baseline Locations",
        "",
        f"- Public release package: `{RELEASE}`",
        f"- PLOS submission working folder: `{SUBMISSION}`",
        f"- Current clean manuscript: `{current_submission}`",
        f"- Current highlighted manuscript: `{highlighted_submission}`",
        f"- Current response letter: `{response_doc}`",
        "",
        "## Current Data Backbone",
        "",
        "- Internal benchmark: strict no-mixed, homology-controlled benchmark derived from BRENDA/UniProt processing.",
        "- External validation panels: strict Seq2Topt-derived novel40 panel, Seq2Topt no-exact panel, and public Topt no-exact panel.",
        "- Current TCBB upgrade should keep `N=66` as the strict low-homology evidence, while using `N=136` and `N=984` to stabilize external evidence and support screening-style analysis.",
        "",
        "### Key Data Artifacts",
        "",
    ]
    for key, row in data_counts.items():
        if row:
            lines.append(
                f"- {key}: `{row['relative_path']}`; rows={row.get('rows','')}; positives={row.get('positive_count','')}; negatives={row.get('negative_count','')}; md5=`{row['md5']}`"
            )
    lines += [
        "",
        "## Key Existing Result Tables",
        "",
        "The following result families already exist and should be treated as historical baseline outputs, not overwritten:",
        "",
    ]
    for name in KEY_RESULTS:
        row = next((r for r in result_rows if r["file_name"] == name), None)
        if row:
            lines.append(f"- `{name}`: rows={row.get('rows','')}; metrics={row.get('metric_columns_detected','')}")
    lines += [
        "",
        "## Baseline Scientific State",
        "",
        "- ESM-2 150M remains the PLOS primary external Figure 5/Table 3 model.",
        "- ESM-2 650M exists as model-scale sensitivity evidence and should not silently redefine the current PLOS Figure 5.",
        "- Regression baseline, mixed-label sensitivity, classifier sensitivity, expanded external validation, and threshold analysis are already completed as PLOS revision outputs.",
        "- For TCBB, the next results should be versioned separately under new phase-specific filenames.",
        "",
        "## Figure Status",
        "",
        "- Current `fig5_real_external_novel40_150m.png` is prediction-derived and is the safest historical PLOS Figure 5 source.",
        "- Current `fig5_real_external_novel40_650m.png` is also prediction-derived, but is a sensitivity figure rather than the PLOS primary figure.",
        "- TCBB Fig. 5 should be redrawn from result CSVs to show ROC/PR across all three external panels.",
        "- TCBB Fig. 6 should be built only after screening/calibration analyses are generated.",
        "- System architecture, data flow/Sankey, UMAP/fusion, and ablation figures are not yet TCBB-ready.",
        "",
        "## Manuscript / Submission State",
        "",
    ]
    for row in sorted(docx_rows, key=lambda r: r["relative_path"]):
        if "PLOS_revision_submission_20260712" in row["relative_path"] or row["file_name"].startswith("ESM-2_revised"):
            info = docx_info(ROOT / row["relative_path"])
            lines.append(
                f"- `{row['relative_path']}`; images={info['images']}; tables={info['tables']}; highlights={info['highlight_elements']}; md5=`{row['md5']}`"
            )
    lines += [
        "",
        "## What Must Not Be Mixed Into Baseline",
        "",
        "- TCBB bootstrap CI across all three external sets has not yet been generated as a unified table.",
        "- Screening EF/Precision@Top-K spiked pools have not yet been generated.",
        "- Probability calibration has not yet been generated.",
        "- Feature fusion, redundancy analysis, group importance, EC stratification, and final six TCBB figures have not yet been generated.",
        "",
        "## Phase 0 Outputs",
        "",
        "- `file_inventory.csv`: complete file-level freeze with size, timestamp, MD5, category, and role.",
        "- `data_inventory.csv`: processed data and feature matrix inventory with detected rows, columns, label columns, counts, and feature dimensions.",
        "- `results_inventory.csv`: result table inventory with detected metric columns and first-row previews.",
        "- `figure_inventory.csv`: figure/docx visual asset inventory with origin status and TCBB action.",
        "- `baseline_key_results.json`: machine-readable snapshots of selected current result tables.",
        "",
        "## Recommended Next Step",
        "",
        "Proceed to Phase 1 by generating `external_bootstrap_ci_all_sets.csv` from the three external result tables, using stratified bootstrap and including ROC-AUC, PR-AUC, MCC, F1, balanced accuracy, precision, recall, and Brier score.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    file_rows = collect_file_inventory()
    data_rows = collect_data_inventory(file_rows)
    result_rows = collect_results_inventory(file_rows)
    figure_rows = collect_figure_inventory(file_rows)

    write_csv(OUT / "file_inventory.csv", file_rows)
    write_csv(OUT / "data_inventory.csv", data_rows)
    write_csv(OUT / "results_inventory.csv", result_rows)
    write_csv(OUT / "figure_inventory.csv", figure_rows)

    key_results = {name: load_result(name) for name in KEY_RESULTS if (RESULT_DIR / name).exists()}
    (OUT / "baseline_key_results.json").write_text(json.dumps(key_results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = make_baseline_summary(file_rows, data_rows, result_rows, figure_rows)
    (OUT / "baseline_summary.md").write_text(summary, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "release_package": str(RELEASE),
        "submission_folder": str(SUBMISSION),
        "outputs": {
            "baseline_summary": str(OUT / "baseline_summary.md"),
            "file_inventory": str(OUT / "file_inventory.csv"),
            "data_inventory": str(OUT / "data_inventory.csv"),
            "results_inventory": str(OUT / "results_inventory.csv"),
            "figure_inventory": str(OUT / "figure_inventory.csv"),
            "baseline_key_results": str(OUT / "baseline_key_results.json"),
        },
    }
    (OUT / "phase0_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
