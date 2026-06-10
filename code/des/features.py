# features.py
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ====== 配置 ======
MODEL_PATH = "fine_tuned_bert"  # 微调后的模型目录
MAX_LENGTH = 128
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 1. 加载模型和tokenizer ======
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertModel.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()


# ====== 2. 特征提取函数 ======
def extract_features(descriptions, batch_size=16):
    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(descriptions), batch_size), desc="Extracting"):
            batch_texts = descriptions[i:i + batch_size]
            encodings = tokenizer(
                batch_texts,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encodings["input_ids"].to(DEVICE)
            attention_mask = encodings["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # 取 [CLS] 向量
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.extend(cls_embeddings)
    return features


# ====== 3. 主函数 ======
if __name__ == "__main__":
    # 加载药物描述数据
    df = pd.read_csv("drug_with_descriptions100.csv")  # 包含 drug_name 和 description

    # 提取特征
    descriptions = df["description"].fillna("").tolist()
    feature_vectors = extract_features(descriptions, BATCH_SIZE)

    # 保存为 Parquet 文件
    table = pa.Table.from_pandas(
        pd.DataFrame({
            "drug_name": df["drug_name"].tolist(),
            "feature_vector": feature_vectors
        })
    )
    pq.write_table(table, "drug_features100.parquet")

    print("特征提取完成，保存到 drug_features100.parquet")
