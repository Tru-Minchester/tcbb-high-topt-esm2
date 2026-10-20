import json
import pandas as pd


def extract_temperature_data(json_file_path, output_csv_path):
    print("正在加载庞大的 BRENDA JSON 文件，请稍候...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        brenda_data = json.load(f)

    enzymes_dict = brenda_data.get('data', {})
    print(f"成功打开 'data' 抽屉，里面共有 {len(enzymes_dict)} 个酶的分类条目！")
    print("开始精准提取单点温度特征并匹配 UniProt ID...")

    extracted_records = []

    for ec_number, ec_data in enzymes_dict.items():
        # 获取这个酶专属的“花名册”
        protein_registry = ec_data.get('protein', {})
        # 获取最适温度数据列表
        temp_records = ec_data.get('temperature_optimum', [])

        for temp_record in temp_records:
            # 1. 提取温度值
            temp_value = temp_record.get('value', '')
            temp_str = str(temp_value).strip()

            # [核心过滤]：跳过空值、区间（带'-'）、以及非纯数字（允许一个小数点）
            if not temp_str or '-' in temp_str or not temp_str.replace('.', '', 1).isdigit():
                continue

                # 2. 拿到内部编号列表 (比如 ['103', '109'])
            protein_ids = temp_record.get('proteins', [])

            # 3. 拿着内部编号，去花名册里“顺藤摸瓜”找真身
            for pid in protein_ids:
                pid_str = str(pid)
                protein_info = protein_registry.get(pid_str, {})

                # 提取真正的 UniProt ID 列表和物种信息
                uniprot_ids = protein_info.get('accessions', [])
                organism = protein_info.get('organism', '')
                source_db = protein_info.get('source', '')

                # 如果这个蛋白质条目没有 UniProt ID，说明无法对应真实序列，丢弃
                if not uniprot_ids:
                    continue

                # 4. 保存每一条干净、有效的映射
                for uid in uniprot_ids:
                    extracted_records.append({
                        'EC_Number': ec_number,
                        'Organism': organism,
                        'Target_Temperature': float(temp_str),
                        'UniProt_Accession': uid,  # 真实的身份代码
                        'Source_DB': source_db  # 比如 swissprot/uniprot
                    })

    if not extracted_records:
        print("提取失败：数据还是空的，请检查逻辑。")
        return

    # 转成 DataFrame
    df = pd.DataFrame(extracted_records)
    print(f"\n大功告成！成功清洗并提取到 {len(df)} 行带有明确 UniProt ID 的温度记录。")

    # 存入 processed 文件夹
    df.to_csv(output_csv_path, index=False)
    print(f"完美！数据已稳稳地保存到: {output_csv_path}")


# --- 运行区 ---
if __name__ == "__main__":
    RAW_JSON = "data_raw/brenda_2026_1.json"
    OUTPUT_CSV = "data_processed/01_clean_temperature_data.csv"

    extract_temperature_data(RAW_JSON, OUTPUT_CSV)