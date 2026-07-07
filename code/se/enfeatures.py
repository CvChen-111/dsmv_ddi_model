"""Extract drug-pair semantic embeddings with the fine-tuned BERT encoder."""
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import sqlite3
from collections import defaultdict

# Configuration
MODEL_PATH = "drug_bert_mlm"
DATA_PATH = "process_event.csv"
DB_PATH = "path/dataset/event.db"
OUTPUT_PATH = "endrug_pair_features.parquet"  # Output file for semantic features.
MAX_LEN = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DrugPairDataset(Dataset):
    """Dataset wrapper for drug-pair texts before BERT encoding."""
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "text": text
        }

def load_drug_pairs():
    """Load drug pairs and construct semantic input text for each pair."""
    try:
        # Load from CSV and keep original drug-pair metadata.
        df = pd.read_csv(DATA_PATH)
        texts = (df["prompt"] + " [SEP] " + df["answer"]).tolist()
        drug_pairs = list(zip(df["drug1"], df["drug2"]))
    except:
        # Load from SQLite database.
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name1, name2, interaction FROM event")
        data = cursor.fetchall()
        texts = [f"{name1} and {name2} [SEP] {interaction}" for name1, name2, interaction in data]
        drug_pairs = [(name1, name2) for name1, name2, _ in data]
        conn.close()
    
    return texts, drug_pairs

def extract_features():
    """Encode all drug pairs and save semantic embeddings to parquet."""
    # Load model and tokenizer.
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    # Load data.
    texts, drug_pairs = load_drug_pairs()
    dataset = DrugPairDataset(texts, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Store all features by drug pair for aggregation.
    feature_dict = defaultdict(list)
    drug_pair_set = set()  # Record all unique drug pairs.

    # Extract features in batches.
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Use the [CLS] token as the sentence representation.
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Retrieve the drug pairs for the current batch.
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + len(batch["text"])
            current_batch_pairs = drug_pairs[start_idx:end_idx]
            
            # Group features by drug pair.
            for pair, feat in zip(current_batch_pairs, cls_embeddings):
                feature_dict[pair].append(feat)
                drug_pair_set.add(pair)

    # Average multiple prompt features for each drug pair.
    aggregated_features = []
    drug1_list = []
    drug2_list = []
    
    for pair in drug_pair_set:
        features = feature_dict[pair]
        # Average pooling can be replaced by another aggregation strategy.
        aggregated_feat = np.mean(features, axis=0)
        aggregated_features.append(aggregated_feat)
        drug1_list.append(pair[0])
        drug2_list.append(pair[1])

    # Convert features to a DataFrame.
    df_features = pd.DataFrame({
        "drug1": drug1_list,
        "drug2": drug2_list,
        "feature": [feat for feat in aggregated_features]
    })

    # Save features in Parquet format.
    df_features.to_parquet(OUTPUT_PATH)
    print(f"Saved AGGREGATED features for {len(df_features)} unique drug pairs to {OUTPUT_PATH}")

    return df_features

if __name__ == "__main__":
    features_df = extract_features()
    # Show one example feature vector.
    print("\nSample aggregated feature vector:")
    print(f"Drug Pair: {features_df.iloc[0]['drug1']} + {features_df.iloc[0]['drug2']}")
    print(f"Feature shape: {features_df.iloc[0]['feature'].shape}")
    print(f"First 10 dims: {features_df.iloc[0]['feature'][:10]}")
