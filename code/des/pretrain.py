"""Fine-tune BERT on single-drug pharmacological descriptions."""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from sklearn.model_selection import train_test_split
import numpy as np

# ====== Configuration ======
MODEL_NAME = 'PATH/bert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5


# ====== 1. Dataset preparation ======
class DrugDescriptionDataset(Dataset):
    """Dataset wrapper for tokenized single-drug descriptions."""
    def __init__(self, descriptions, tokenizer, max_length):
        self.descriptions = descriptions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = str(self.descriptions[idx])

        # Tokenize the description text.
        encoding = self.tokenizer(
            description,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Create masked-language-model labels.
        input_ids = encoding['input_ids'].squeeze()
        labels = input_ids.clone()

        # Randomly mask 15% of tokens.
        mask_indices = torch.rand(input_ids.shape) < 0.15
        # Avoid masking special tokens.
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


# ====== 2. Load data ======
def load_data(csv_file):
    """Load drug descriptions from CSV and split train/validation subsets."""
    df = pd.read_csv(csv_file)
    descriptions = df['description'].dropna().tolist()
    return descriptions


# ====== 3. Fine-tuning function ======
def fine_tune_bert(train_descriptions, val_descriptions=None):
    """Fine-tune BERT on description texts and save the best model."""
    # Initialize tokenizer and model.
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForMaskedLM.from_pretrained(MODEL_NAME)

    # Create datasets.
    train_dataset = DrugDescriptionDataset(train_descriptions, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if val_descriptions:
        val_dataset = DrugDescriptionDataset(val_descriptions, tokenizer, MAX_LENGTH)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Optimizer.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop.
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

    # Save the fine-tuned model.
    model.save_pretrained('fine_tuned_bert')
    tokenizer.save_pretrained('fine_tuned_bert')
    print("Done! Model seved to fine_tuned_bert")

    return model, tokenizer


# ====== Main routine ======
if __name__ == "__main__":
    # Load data.
    descriptions = load_data('drug_with_descriptions100.csv')

    # Split training and validation subsets.
    train_descriptions, val_descriptions = train_test_split(
        descriptions, test_size=0.2, random_state=42
    )

    print(f"train_num: {len(train_descriptions)}")
    print(f"validation_num: {len(val_descriptions)}")

    # Start fine-tuning.
    fine_tune_bert(train_descriptions, val_descriptions)
