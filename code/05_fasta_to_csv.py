import pandas as pd


def fasta_to_csv(input_fasta, output_csv):
    print("正在解析聚类后的 FASTA 文件...")
    records = []

    with open(input_fasta, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 2):
        header = lines[i].strip()
        seq = lines[i + 1].strip()

        if header.startswith('>'):
            # 还原我们之前藏在 Header 里的 ID 和温度
            # 格式是：>UniProtID_TEMP_温度
            parts = header[1:].split('_TEMP_')
            if len(parts) == 2:
                uid = parts[0]
                temp = float(parts[1])

                records.append({
                    'UniProt_Accession': uid,
                    'Target_Temperature': temp,
                    'Protein_Sequence': seq
                })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"转换完成！最终获得 {len(df)} 条低同源性数据，已保存至: {output_csv}")


if __name__ == "__main__":
    # 记得把你从 Colab 下载的文件放进这个路径
    INPUT_FASTA = "data_processed/04_cdhit40_clustered.fasta"
    OUTPUT_CSV = "data_processed/05_final_low_homology_dataset.csv"

    fasta_to_csv(INPUT_FASTA, OUTPUT_CSV)