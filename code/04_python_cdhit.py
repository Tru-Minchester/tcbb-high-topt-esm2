import pandas as pd
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm


def python_greedy_clustering(input_csv, output_csv, identity_threshold=0.40):
    print("正在加载完整数据集...")
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=['Protein_Sequence'])

    # 1. 核心逻辑：按序列长度从长到短排序 (模仿 CD-HIT)
    df['Seq_Length'] = df['Protein_Sequence'].apply(len)
    df = df.sort_values(by='Seq_Length', ascending=False).reset_index(drop=True)

    print(f"准备对 {len(df)} 条序列进行严格的 {identity_threshold * 100}% 低同源聚类...")

    representative_seeds = []  # 用于保存每个簇的“代表”（最长序列）

    # 2. 贪婪聚类循环
    for index, row in tqdm(df.iterrows(), total=len(df), desc="聚类进度"):
        current_seq = row['Protein_Sequence']
        is_redundant = False

        # 和目前已经建立的“种子库”进行对比
        for seed in representative_seeds:
            # rapidfuzz 的 normalized_similarity 计算速度极快，返回 0.0 ~ 1.0 的相似度
            # 它在本质上等价于全局序列比对（Edit Distance Identity）
            sim = Levenshtein.normalized_similarity(current_seq, seed['Protein_Sequence'])

            if sim >= identity_threshold:
                # 相似度达到阈值，说明是高度同源的“废品”，标记为冗余并跳出
                is_redundant = True
                break

        # 如果和所有已有的种子都不怎么像，说明它是一个全新的家族，加入种子库
        if not is_redundant:
            representative_seeds.append(row)

    # 3. 聚类完成，保存结果
    clustered_df = pd.DataFrame(representative_seeds)
    # 删掉刚才用来排序的辅助列
    clustered_df = clustered_df.drop(columns=['Seq_Length'])

    print(
        f"\n大功告成！从 {len(df)} 条原始序列中，提炼出了 {len(clustered_df)} 条互相之间相似度均低于 40% 的独立代表序列。")

    clustered_df.to_csv(output_csv, index=False)
    print(f"极度干净的 Low-homology 数据集已保存至: {output_csv}")


# --- 运行区 ---
if __name__ == "__main__":
    INPUT_CSV = "data_processed/02_main_dataset_with_sequences.csv"
    OUTPUT_CSV = "data_processed/04_final_low_homology_dataset.csv"

    # 40% 的严格阈值
    python_greedy_clustering(INPUT_CSV, OUTPUT_CSV, identity_threshold=0.40)