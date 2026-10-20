import pandas as pd
import requests
import time
import os
from tqdm import tqdm


def fetch_uniprot_sequences(input_csv, output_csv, cache_file):
    print("正在加载第一步洗好的温度数据...")
    df = pd.read_csv(input_csv)

    # 拿到所有的 UniProt ID，并去重（避免同一个酶重复下载）
    unique_ids = df['UniProt_Accession'].dropna().unique()
    print(f"共有 {len(df)} 行数据，其中包含 {len(unique_ids)} 个不重复的 UniProt ID。")

    # 加载本地缓存（断点续传的核心）
    sequence_dict = {}
    if os.path.exists(cache_file):
        print(f"发现本地缓存文件 '{cache_file}'，正在加载...")
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    sequence_dict[parts[0]] = parts[1]
        print(f"已恢复 {len(sequence_dict)} 条历史序列记录。")

    # 找出还需要下载的 ID
    ids_to_fetch = [uid for uid in unique_ids if uid not in sequence_dict]
    print(f"本次还需要向 UniProt 请求 {len(ids_to_fetch)} 条新序列。")
    print("开始排队向 UniProt 进货 (可能需要几十分钟，中途随时可以按 Ctrl+C 中断，下次运行会接上)...\n")

    # 打开缓存文件，准备追加写入新下载的数据
    with open(cache_file, 'a', encoding='utf-8') as cache_f:
        # 使用 tqdm 弄一个漂亮的进度条
        for uid in tqdm(ids_to_fetch, desc="拉取序列进度"):
            url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"

            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # 解析 FASTA 格式：第一行是表头(>开头)，后面的是序列
                    lines = response.text.strip().split('\n')
                    sequence = "".join(lines[1:])  # 把序列拼成一长串没有换行的字符串

                    if sequence:
                        sequence_dict[uid] = sequence
                        # 成功一条就立刻存进本地文本，防止断电白跑
                        cache_f.write(f"{uid}\t{sequence}\n")
                        cache_f.flush()

                # UniProt 官方要求：不要频繁轰炸服务器，礼貌停顿一下
                time.sleep(0.1)

            except Exception as e:
                print(f"\n下载 {uid} 时遇到网络波动: {e}")
                time.sleep(2)  # 遇到错误休息2秒继续

    print("\n\n序列进货完成！现在开始把序列拼接到你的原始表格里...")

    # 使用 Pandas 的 map 功能，根据 UniProt_Accession 匹配出对应的序列，加到新的一列
    df['Protein_Sequence'] = df['UniProt_Accession'].map(sequence_dict)

    # 清洗：有些 ID 已经被 UniProt 官方废弃了，抓不到序列的空行我们直接丢掉
    initial_len = len(df)
    df = df.dropna(subset=['Protein_Sequence'])
    df = df[df['Protein_Sequence'] != ""]  # 去掉空字符串

    print(f"剔除了 {initial_len - len(df)} 个序列获取失败的废弃条目。")

    # 保存包含序列的终极完整表格
    df.to_csv(output_csv, index=False)
    print(f"完美！包含特征序列的最终数据集已保存至: {output_csv}")
    print(f"最终有效数据量: {len(df)} 条")


# --- 运行区 ---
if __name__ == "__main__":
    INPUT_CSV = "data_processed/01_clean_temperature_data.csv"
    OUTPUT_CSV = "data_processed/02_main_dataset_with_sequences.csv"
    CACHE_FILE = "data_raw/uniprot_sequence_cache.txt"

    fetch_uniprot_sequences(INPUT_CSV, OUTPUT_CSV, CACHE_FILE)