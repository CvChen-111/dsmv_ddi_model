"""Extract single-drug description embeddings with a BERT encoder."""
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ====== Configuration ======
MODEL_PATH = "fine_tuned_bert"  # Fine-tuned model directory
MAX_LENGTH = 128
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 1. Load model and tokenizer ======
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertModel.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()


# ====== 2. Feature extraction ======
def extract_features(descriptions, batch_size=16):
    """Encode drug descriptions in batches and return embedding vectors."""
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
            # Use the [CLS] vector as the sentence representation.
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.extend(cls_embeddings)
    return features


# ====== 3. Main routine ======
if __name__ == "__main__":
    # Load drug description data.
    df = pd.read_csv("drug_with_descriptions100.csv")  # Contains drug_name and description.

    # Extract description embeddings.
    descriptions = df["description"].fillna("").tolist()
    feature_vectors = extract_features(descriptions, BATCH_SIZE)

    # Save features as a Parquet file.
    table = pa.Table.from_pandas(
        pd.DataFrame({
            "drug_name": df["drug_name"].tolist(),
            "feature_vector": feature_vectors
        })
    )
    pq.write_table(table, "drug_features100.parquet")

    print("Feature extraction completed, saved to drug_features100.parquet")
