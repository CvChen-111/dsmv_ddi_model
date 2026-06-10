import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 配置
MODEL_PATH = "/data/cclsol/cfn/MF-bert-0.959/CD/code/bert-base-uncased"
DATA_PATH = "process_event.csv"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# 自定义 Dataset
class DrugPairDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 拼接药物对文本（prompt + answer）
        text = f"{self.texts[idx]['prompt']} [SEP] {self.texts[idx]['answer']}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True
        )
        
        # 随机遮蔽15%的token（MLM任务）
        input_ids = encoding["input_ids"].clone()
        labels = input_ids.clone()
        
        # 创建遮蔽掩码（忽略特殊token）
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = encoding["special_tokens_mask"].bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 只计算遮蔽位置的loss
        
        # 80%概率替换为[MASK]，10%随机token，10%保持原词
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }

# 加载数据
def load_data():
    df = pd.read_csv(DATA_PATH)
    texts = []
    for _, row in df.iterrows():
        texts.append({
            "prompt": row["prompt"],
            "answer": row["answer"]
        })
    return texts

# 训练函数
def train():
    # 初始化
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForMaskedLM.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 数据加载
    texts = load_data()
    train_size = int(0.9 * len(texts))
    train_texts, val_texts = texts[:train_size], texts[train_size:]
    
    train_dataset = DrugPairDataset(train_texts, tokenizer, MAX_LEN)
    val_dataset = DrugPairDataset(val_texts, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device)
            }
            
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "labels": batch["labels"].to(device)
                }
                outputs = model(**inputs)
                val_loss += outputs.loss.item()
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
    
    # 保存模型
    model.save_pretrained("drug_bert_mlm")
    tokenizer.save_pretrained("drug_bert_mlm")
    print("Model saved for feature extraction.")

if __name__ == "__main__":
    train()