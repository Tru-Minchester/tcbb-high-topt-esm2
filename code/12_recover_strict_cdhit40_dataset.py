from pathlib import Path
from typing import Dict, List, Tuple
import re

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "11_unique_clean_sequences.csv"
REP_FASTA = BASE_DIR / "12_cdhit40_strict.fasta"
CLSTR_FILE = BASE_DIR / "12_cdhit40_strict.fasta.clstr"

REP_OUT = BASE_DIR / "12_cdhit40_strict_dataset.csv"
CLUSTER_MEMBER_OUT = BASE_DIR / "12_cdhit40_cluster_members.csv"
CLUSTER_SUMMARY_OUT = BASE_DIR / "12_cdhit40_cluster_summary.csv"
SUMMARY_TXT = BASE_DIR / "12_recover_strict_cdhit40_summary.txt"

SEQ_COL = "Protein_Sequence"
HASH_COL = "Sequence_Hash"
LABEL_COL = "Binary_Label"
LEN_COL = "Sequence_Length"
PREFIX_LEN = 19


def parse_fasta_headers(fasta_path: Path) -> List[str]:
    headers = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                headers.append(line[1:])
    return headers


def hash_from_header(header: str) -> str:
    return header.split("|")[0].strip()


def parse_clstr(clstr_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cluster_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    current_cluster = None
    current_members: List[Dict] = []

    with open(clstr_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">Cluster"):
                if current_cluster is not None and current_members:
                    summary_rows.append(summarize_cluster(current_cluster, current_members))
                    cluster_rows.extend(current_members)
                current_cluster = int(line.replace(">Cluster", "").strip())
                current_members = []
                continue

            member = parse_clstr_member_line(line, current_cluster)
            current_members.append(member)

    if current_cluster is not None and current_members:
        summary_rows.append(summarize_cluster(current_cluster, current_members))
        cluster_rows.extend(current_members)

    return pd.DataFrame(cluster_rows), pd.DataFrame(summary_rows)


def parse_clstr_member_line(line: str, cluster_id: int) -> Dict:
    # Example:
    # 0\t7096aa, >c0227764f2991b4d1ce... *
    # 1\t7073aa, >902414615cf6c4ca702... at 78.89%
    m = re.match(r'^(\d+)\s+(\d+)aa,\s+>([^.]+)\.\.\.\s*(\*|at\s+.+)?$', line)
    if not m:
        raise ValueError(f"Unable to parse .clstr line: {line}")

    member_index = int(m.group(1))
    seq_len = int(m.group(2))
    hash_prefix = m.group(3).strip()
    tail = (m.group(4) or "").strip()

    is_representative = int(tail == "*")
    identity_to_rep = None if is_representative else (tail.replace("at ", "").strip() or None)

    return {
        "Cluster_ID": cluster_id,
        "Member_Index": member_index,
        "Member_Header": hash_prefix,
        "Sequence_Hash_Prefix": hash_prefix,
        "CLSTR_Length": seq_len,
        "Is_Representative": is_representative,
        "Identity_To_Representative": identity_to_rep,
    }


def summarize_cluster(cluster_id: int, members: List[Dict]) -> Dict:
    sizes = [m["CLSTR_Length"] for m in members]
    rep_rows = [m for m in members if m["Is_Representative"] == 1]
    rep_hash_prefix = rep_rows[0]["Sequence_Hash_Prefix"] if rep_rows else ""
    return {
        "Cluster_ID": cluster_id,
        "Cluster_Size": len(members),
        "Representative_Hash_Prefix": rep_hash_prefix,
        "Min_Length": min(sizes),
        "Max_Length": max(sizes),
        "Length_Ratio_Max_Min": (max(sizes) / min(sizes)) if min(sizes) > 0 else None,
    }


def main():
    print("=" * 80)
    print("Step 12: recover strict CD-HIT 40% representative dataset")
    print("=" * 80)
    print(f"[INFO] Working directory: {Path.cwd()}")
    print(f"[INFO] Script directory: {BASE_DIR}")

    for p in [INPUT_CSV, REP_FASTA, CLSTR_FILE]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    source_df = pd.read_csv(INPUT_CSV)
    if HASH_COL not in source_df.columns:
        raise ValueError(f"Missing required column in source table: {HASH_COL}")

    source_df = source_df.drop_duplicates(subset=[HASH_COL]).copy()
    source_df[HASH_COL] = source_df[HASH_COL].astype(str)
    source_df["Sequence_Hash_Prefix"] = source_df[HASH_COL].str[:PREFIX_LEN]

    if source_df["Sequence_Hash_Prefix"].duplicated().any():
        dup_prefixes = source_df.loc[
            source_df["Sequence_Hash_Prefix"].duplicated(),
            "Sequence_Hash_Prefix"
        ].unique()[:10]
        raise ValueError(f"Non-unique hash prefixes detected: {dup_prefixes}")

    prefix_to_hash = dict(zip(source_df["Sequence_Hash_Prefix"], source_df[HASH_COL]))

    headers = parse_fasta_headers(REP_FASTA)
    rep_hashes = [hash_from_header(h) for h in headers]
    rep_hashes_unique = list(dict.fromkeys(rep_hashes))

    rep_df = source_df[source_df[HASH_COL].isin(rep_hashes_unique)].copy()
    rep_df = rep_df.set_index(HASH_COL).loc[rep_hashes_unique].reset_index()

    cluster_members_df, cluster_summary_df = parse_clstr(CLSTR_FILE)

    cluster_members_df[HASH_COL] = cluster_members_df["Sequence_Hash_Prefix"].map(prefix_to_hash)
    missing_hashes = int(cluster_members_df[HASH_COL].isna().sum())
    if missing_hashes > 0:
        print(f"[WARN] Unmapped .clstr hash prefixes: {missing_hashes}")

    merge_cols = list(source_df.columns)
    cluster_members_df = cluster_members_df.merge(
        source_df[merge_cols],
        on=HASH_COL,
        how="left"
    )

    rep_cluster_df = cluster_members_df[cluster_members_df["Is_Representative"] == 1][
        ["Cluster_ID", HASH_COL]
    ].rename(columns={HASH_COL: "Representative_Hash"})

    cluster_summary_df = cluster_summary_df.drop(
        columns=["Representative_Hash_Prefix"],
        errors="ignore"
    )
    cluster_summary_df = cluster_summary_df.merge(
        rep_cluster_df,
        on="Cluster_ID",
        how="left"
    )

    if LABEL_COL in cluster_members_df.columns:
        label_stats = cluster_members_df.groupby("Cluster_ID")[LABEL_COL].agg([
            ("Positives_In_Cluster", lambda s: int((s == 1).sum())),
            ("Negatives_In_Cluster", lambda s: int((s == 0).sum())),
        ]).reset_index()
        label_stats["Mixed_Label_Cluster"] = (
            (label_stats["Positives_In_Cluster"] > 0) &
            (label_stats["Negatives_In_Cluster"] > 0)
        ).astype(int)

        cluster_summary_df = cluster_summary_df.merge(
            label_stats,
            on="Cluster_ID",
            how="left"
        )

    rep_df = rep_df.merge(
        cluster_summary_df[["Cluster_ID", "Representative_Hash", "Cluster_Size"]],
        left_on=HASH_COL,
        right_on="Representative_Hash",
        how="left"
    ).drop(columns=["Representative_Hash"])

    rep_df.to_csv(REP_OUT, index=False)
    cluster_members_df.to_csv(CLUSTER_MEMBER_OUT, index=False)
    cluster_summary_df.to_csv(CLUSTER_SUMMARY_OUT, index=False)

    pos = int((rep_df[LABEL_COL] == 1).sum()) if LABEL_COL in rep_df.columns else -1
    neg = int((rep_df[LABEL_COL] == 0).sum()) if LABEL_COL in rep_df.columns else -1
    ratio = (neg / pos) if pos > 0 else None
    mixed_clusters = int(cluster_summary_df["Mixed_Label_Cluster"].sum()) if "Mixed_Label_Cluster" in cluster_summary_df.columns else -1
    large_ratio_clusters = int((cluster_summary_df["Length_Ratio_Max_Min"] > 10).sum()) if len(cluster_summary_df) else 0

    summary_lines = [
        "Step 12 summary",
        f"Input clean sequences: {len(source_df)}",
        f"Representative FASTA entries: {len(rep_hashes_unique)}",
        f"Recovered representative rows: {len(rep_df)}",
        f"Clusters parsed from .clstr: {len(cluster_summary_df)}",
        f"Unmapped .clstr hash prefixes: {missing_hashes}",
        f"Positive representatives: {pos}",
        f"Negative representatives: {neg}",
        f"Positive:negative ratio = 1:{ratio:.2f}" if ratio is not None else "No positive samples",
        f"Mixed-label clusters: {mixed_clusters}",
        f"Clusters with length ratio > 10: {large_ratio_clusters}",
        f"Saved representative dataset: {REP_OUT.name}",
        f"Saved cluster members: {CLUSTER_MEMBER_OUT.name}",
        f"Saved cluster summary: {CLUSTER_SUMMARY_OUT.name}",
    ]
    SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

    print("-" * 80)
    print(f"[OK] Saved representative dataset: {REP_OUT}")
    print(f"[OK] Saved cluster member table: {CLUSTER_MEMBER_OUT}")
    print(f"[OK] Saved cluster summary table: {CLUSTER_SUMMARY_OUT}")
    print(f"[OK] Saved summary: {SUMMARY_TXT}")
    print("-" * 80)
    print(f"[INFO] Representative rows: {len(rep_df)}")
    print(f"[INFO] Cluster count: {len(cluster_summary_df)}")
    if pos >= 0:
        print(f"[INFO] Positives: {pos}")
        print(f"[INFO] Negatives: {neg}")
        if ratio is not None:
            print(f"[INFO] Positive:negative ratio = 1:{ratio:.2f}")
    if mixed_clusters >= 0:
        print(f"[INFO] Mixed-label clusters: {mixed_clusters}")
    print(f"[INFO] Clusters with length ratio > 10: {large_ratio_clusters}")
    print("=" * 80)
    print("[DONE] Strict CD-HIT representative recovery finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()