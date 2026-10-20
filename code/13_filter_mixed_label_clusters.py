from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_DATASET = BASE_DIR / "12_cdhit40_strict_dataset.csv"
INPUT_CLUSTER_SUMMARY = BASE_DIR / "12_cdhit40_cluster_summary.csv"

STRICT_OUT = BASE_DIR / "13_cdhit40_strict_nomixed_dataset.csv"
REMOVED_OUT = BASE_DIR / "13_mixed_label_representatives_removed.csv"
SUMMARY_TXT = BASE_DIR / "13_filter_mixed_label_clusters_summary.txt"

LABEL_COL = "Binary_Label"
CLUSTER_ID_COL = "Cluster_ID"
MIXED_COL = "Mixed_Label_Cluster"


def main():
    print("=" * 80)
    print("Step 13: remove mixed-label clusters from strict CD-HIT dataset")
    print("=" * 80)
    print(f"[INFO] Working directory: {Path.cwd()}")
    print(f"[INFO] Script directory: {BASE_DIR}")

    for p in [INPUT_DATASET, INPUT_CLUSTER_SUMMARY]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    df = pd.read_csv(INPUT_DATASET)
    cluster_df = pd.read_csv(INPUT_CLUSTER_SUMMARY)

    required_dataset_cols = [CLUSTER_ID_COL, LABEL_COL]
    missing_dataset = [c for c in required_dataset_cols if c not in df.columns]
    if missing_dataset:
        raise ValueError(f"Missing columns in dataset: {missing_dataset}")

    if MIXED_COL not in cluster_df.columns or CLUSTER_ID_COL not in cluster_df.columns:
        raise ValueError(f"Cluster summary must contain {CLUSTER_ID_COL} and {MIXED_COL}")

    mixed_cluster_ids = set(
        cluster_df.loc[cluster_df[MIXED_COL] == 1, CLUSTER_ID_COL].tolist()
    )

    removed_df = df[df[CLUSTER_ID_COL].isin(mixed_cluster_ids)].copy()
    strict_df = df[~df[CLUSTER_ID_COL].isin(mixed_cluster_ids)].copy()

    strict_df.to_csv(STRICT_OUT, index=False)
    removed_df.to_csv(REMOVED_OUT, index=False)

    pos = int((strict_df[LABEL_COL] == 1).sum())
    neg = int((strict_df[LABEL_COL] == 0).sum())
    ratio = (neg / pos) if pos > 0 else float("inf")

    removed_pos = int((removed_df[LABEL_COL] == 1).sum()) if len(removed_df) else 0
    removed_neg = int((removed_df[LABEL_COL] == 0).sum()) if len(removed_df) else 0

    summary_lines = [
        "Step 13 summary",
        f"Input representatives: {len(df)}",
        f"Mixed-label clusters removed: {len(mixed_cluster_ids)}",
        f"Representatives removed: {len(removed_df)}",
        f"Remaining strict representatives: {len(strict_df)}",
        f"Remaining positives: {pos}",
        f"Remaining negatives: {neg}",
        f"Remaining positive:negative ratio = 1:{ratio:.2f}" if pos > 0 else "No positive samples remaining",
        f"Removed positives: {removed_pos}",
        f"Removed negatives: {removed_neg}",
        f"Saved strict no-mixed dataset: {STRICT_OUT.name}",
        f"Saved removed representatives: {REMOVED_OUT.name}",
    ]
    SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

    print("-" * 80)
    print(f"[OK] Saved strict no-mixed dataset: {STRICT_OUT}")
    print(f"[OK] Saved removed representatives: {REMOVED_OUT}")
    print(f"[OK] Saved summary: {SUMMARY_TXT}")
    print("-" * 80)
    print(f"[INFO] Mixed-label clusters removed: {len(mixed_cluster_ids)}")
    print(f"[INFO] Representatives removed: {len(removed_df)}")
    print(f"[INFO] Remaining strict representatives: {len(strict_df)}")
    print(f"[INFO] Remaining positives: {pos}")
    print(f"[INFO] Remaining negatives: {neg}")
    if pos > 0:
        print(f"[INFO] Remaining positive:negative ratio = 1:{ratio:.2f}")
    print("=" * 80)
    print("[DONE] Mixed-label cluster filtering finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
