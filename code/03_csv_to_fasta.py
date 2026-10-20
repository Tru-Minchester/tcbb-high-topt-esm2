import pandas as pd


def csv_to_fasta(input_csv, output_fasta):
    print("正在加载包含序列的终极完整表格...")
    df = pd.read_csv(input_csv)

    # 丢掉那些序列为空的行（保险起见）
    df = df.dropna(subset=['Protein_Sequence'])

    print(f"准备转换 {len(df)} 条序列为 FASTA 格式...")

    with open(output_fasta, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            # 提取信息
            uid = row['UniProt_Accession']
            temp = row['Target_Temperature']
            seq = row['Protein_Sequence']

            # 格式化为 FASTA 头部。
            # 这里的巧妙设计是：我们把温度标签偷偷藏在 ID 后面，
            # 这样等 CD-HIT 去重完毕后，我们还能把温度找回来！
            header = f">{uid}_TEMP_{temp}"

            # 写入文件
            f.write(f"{header}\n{seq}\n")

    print(f"\n转换完成！标准的生信 FASTA 文件已生成：{output_fasta}")


# --- 运行区 ---
if __name__ == "__main__":
    INPUT_CSV = "data_processed/02_main_dataset_with_sequences.csv"
    OUTPUT_FASTA = "data_processed/03_unclustered_sequences.fasta"

    csv_to_fasta(INPUT_CSV, OUTPUT_FASTA)