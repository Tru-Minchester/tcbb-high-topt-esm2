import os
import pandas as pd

# =========================================================
# 1. 自动找文件：优先根目录，其次 data_processed 目录
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ORIGINAL_CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "02_main_dataset_with_sequences.csv"),
    os.path.join(BASE_DIR, "data_processed", "02_main_dataset_with_sequences.csv"),
]

CLUSTERED_FASTA_CANDIDATES = [
    os.path.join(BASE_DIR, "04_cdhit40_clustered.fasta"),
    os.path.join(BASE_DIR, "data_processed", "04_cdhit40_clustered.fasta"),
]

OUTPUT_CSV = os.path.join(BASE_DIR, "05_main_dataset_cdhit40.csv")

ACC_COL = "UniProt_Accession"
TEMP_COL = "Target_Temperature"
SEQ_COL = "Protein_Sequence"


def find_existing_file(candidates, file_desc="文件"):
    for path in candidates:
        if os.path.exists(path):
            print(f"[INFO] 找到{file_desc}: {path}")
            return path

    print(f"[ERROR] 未找到{file_desc}。尝试过这些路径：")
    for p in candidates:
        print("   -", p)
    raise FileNotFoundError(f"未找到{file_desc}")


def read_fasta_headers_and_sequences(fasta_path):
    records = []
    current_header = None
    current_seq = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    records.append((current_header, "".join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_header is not None:
            records.append((current_header, "".join(current_seq)))

    return records


def parse_header(header):
    """
    你的 header 格式示例：
    P39462_TEMP_80.0
    """
    if "_TEMP_" not in header:
        raise ValueError(f"无法解析 FASTA header: {header}")
    acc, temp = header.split("_TEMP_")
    return acc.strip(), float(temp)


def main():
    # 自动定位文件
    input_original_csv = find_existing_file(ORIGINAL_CSV_CANDIDATES, "原始主数据 CSV")
    input_clustered_fasta = find_existing_file(CLUSTERED_FASTA_CANDIDATES, "CD-HIT 聚类 FASTA")

    # 读取原始数据
    df = pd.read_csv(input_original_csv)

    required_cols = [ACC_COL, TEMP_COL, SEQ_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"原始 CSV 缺少列: {missing}")

    df[ACC_COL] = df[ACC_COL].astype(str).str.strip()
    df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors="coerce")
    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()

    # 读 clustered fasta
    fasta_records = read_fasta_headers_and_sequences(input_clustered_fasta)

    recovered_rows = []
    not_found = 0

    for header, seq in fasta_records:
        acc, temp = parse_header(header)

        # 严格匹配 accession + temp + sequence
        sub = df[
            (df[ACC_COL] == acc) &
            (df[TEMP_COL] == temp) &
            (df[SEQ_COL] == seq)
        ].copy()

        # 放宽匹配 accession + temp
        if len(sub) == 0:
            sub = df[
                (df[ACC_COL] == acc) &
                (df[TEMP_COL] == temp)
            ].copy()

        if len(sub) == 0:
            not_found += 1
            recovered_rows.append({
                "UniProt_Accession": acc,
                "Target_Temperature": temp,
                "Protein_Sequence": seq,
                "Recovered_From_FASTA_Only": True
            })
        else:
            row = sub.iloc[0].to_dict()
            row["Recovered_From_FASTA_Only"] = False
            recovered_rows.append(row)

    out_df = pd.DataFrame(recovered_rows)

    # 去重
    out_df = out_df.drop_duplicates(
        subset=["UniProt_Accession", "Target_Temperature", "Protein_Sequence"]
    ).copy()

    out_df.to_csv(OUTPUT_CSV, index=False)

    print("=" * 70)
    print(f"[OK] 输出文件: {OUTPUT_CSV}")
    print(f"[OK] 最终条目数: {len(out_df)}")
    print(f"[OK] 仅靠 FASTA 恢复的条目数: {not_found}")
    print("=" * 70)

    # 统计类别分布
    out_df["Label"] = (pd.to_numeric(out_df["Target_Temperature"], errors="coerce") >= 60.0).astype(int)
    pos = int(out_df["Label"].sum())
    neg = int(len(out_df) - pos)

    print(f"Positive (>=60°C): {pos}")
    print(f"Negative (<60°C): {neg}")
    if pos > 0:
        print(f"Positive:Negative = 1:{neg / pos:.2f}")


if __name__ == "__main__":
    main()