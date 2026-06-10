import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from sklearn.model_selection import train_test_split
import numpy as np

# ====== 配置参数 ======
MODEL_NAME = '/data/cclsol/cfn/MF-bert-0.959/CD/code/bert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5


# ====== 1. 数据准备 ======
class DrugDescriptionDataset(Dataset):
    def __init__(self, descriptions, tokenizer, max_length):
        self.descriptions = descriptions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = str(self.descriptions[idx])

        # 对描述进行编码
        encoding = self.tokenizer(
            description,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 创建掩码语言模型标签
        input_ids = encoding['input_ids'].squeeze()
        labels = input_ids.clone()

        # 随机掩码15%的token
        mask_indices = torch.rand(input_ids.shape) < 0.15
        # 确保特殊token不被掩码
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            input_ids, already_has_special_tokens=True
        )
        mask_indices = mask_indices & (torch.tensor(special_tokens_mask) == 0)

        input_ids[mask_indices] = self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }


# ====== 2. 加载数据 ======
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    descriptions = df['description'].dropna().tolist()
    return descriptions


# ====== 3. 微调函数 ======
def fine_tune_bert(train_descriptions, val_descriptions=None):
    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForMaskedLM.from_pretrained(MODEL_NAME)

    # 创建数据集
    train_dataset = DrugDescriptionDataset(train_descriptions, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if val_descriptions:
        val_dataset = DrugDescriptionDataset(val_descriptions, tokenizer, MAX_LENGTH)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 训练循环
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}')

    # 保存微调后的模型
    model.save_pretrained('fine_tuned_bert')
    tokenizer.save_pretrained('fine_tuned_bert')
    print("微调完成，模型已保存到 fine_tuned_bert")

    return model, tokenizer


# ====== 主函数 ======
if __name__ == "__main__":
    # 加载数据
    descriptions = load_data('drug_with_descriptions100.csv')

    # 分割训练集和验证集
    train_descriptions, val_descriptions = train_test_split(
        descriptions, test_size=0.2, random_state=42
    )

    print(f"训练样本数: {len(train_descriptions)}")
    print(f"验证样本数: {len(val_descriptions)}")

    # 开始微调
    fine_tune_bert(train_descriptions, val_descriptions)