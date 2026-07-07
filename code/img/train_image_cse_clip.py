#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and use the molecular image encoder with CSE supervision."""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import argparse


# =========================
# Dataset
# =========================

class DrugImageDataset(Dataset):
    """Dataset that pairs drug images with precomputed CSE embeddings."""

    def __init__(self, img_dir, cse_feat_path, transform):

        self.img_dir = img_dir
        self.transform = transform

        self.cse_feat = torch.load(cse_feat_path)

        self.samples = []

        for file in os.listdir(img_dir):

            if not file.endswith(".png"):
                continue

            m = re.match(r"(DB\d+)_view[1-4]\.png", file)
            if m is None:
                continue

            drug_id = m.group(1)

            if drug_id not in self.cse_feat:
                continue

            self.samples.append((drug_id, os.path.join(img_dir,file)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        drug_id, path = self.samples[idx]

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        cse = self.cse_feat[drug_id].float()

        return img, cse


# =========================
# CLIP Model
# =========================

class ImageEncoder(nn.Module):
    """ResNet-based image encoder for molecular renderings."""

    def __init__(self, embed_dim=256):

        super().__init__()

        backbone = models.resnet50(weights=None)

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.proj = nn.Linear(feat_dim, embed_dim)

    def forward(self, x):

        x = self.backbone(x)
        x = self.proj(x)

        return F.normalize(x, dim=1)


class CSEEncoder(nn.Module):
    """Projection head for CSE features used in contrastive training."""

    def __init__(self, input_dim, embed_dim=256):

        super().__init__()

        self.fc = nn.Linear(input_dim, embed_dim)

    def forward(self, x):

        x = self.fc(x)
        return F.normalize(x, dim=1)


class CLIP(nn.Module):
    """Contrastive model aligning molecular images and CSE features."""

    def __init__(self, cse_dim):

        super().__init__()

        self.image_encoder = ImageEncoder()
        self.cse_encoder = CSEEncoder(cse_dim)

        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, img, cse):

        img_feat = self.image_encoder(img)
        cse_feat = self.cse_encoder(cse)

        logits = img_feat @ cse_feat.T / self.temperature.exp()

        return logits


# =========================
# Train
# =========================

def train(args):
    """Train the image encoder using image-CSE contrastive alignment."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    dataset = DrugImageDataset(
        args.img_dir,
        args.cse_feat,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    cse_dim = next(iter(dataset))[1].shape[0]

    model = CLIP(cse_dim).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    for epoch in range(args.epochs):

        total_loss = 0

        for img,cse in tqdm(loader):

            img = img.to(device)
            cse = cse.to(device)

            logits = model(img,cse)

            labels = torch.arange(img.size(0)).to(device)

            loss_i = F.cross_entropy(logits,labels)
            loss_t = F.cross_entropy(logits.T,labels)

            loss = (loss_i + loss_t)/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch",epoch,"Loss",total_loss/len(loader))

    torch.save(model.image_encoder.state_dict(),"image_encoder.pt")


# =========================
# Extract features
# =========================

def extract(img_dir, encoder_path, output):
    """Extract image embeddings with a trained image encoder."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ImageEncoder()
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.to(device)
    encoder.eval()

    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    feats = {}

    files = os.listdir(img_dir)

    for file in tqdm(files):

        if not file.endswith(".png"):
            continue

        m = re.match(r"(DB\d+)_view[1-4]\.png", file)

        if m is None:
            continue

        drug = m.group(1)

        img = Image.open(os.path.join(img_dir,file)).convert("RGB")

        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            f = encoder(img).cpu().squeeze()

        feats.setdefault(drug, []).append(f)

    for drug in feats:

        feats[drug] = torch.stack(feats[drug]).mean(0)

    torch.save(feats, output)


# =========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--img_dir")
    parser.add_argument("--cse_feat")
    parser.add_argument("--epochs",type=int,default=20)

    args = parser.parse_args()

    train(args)

    extract(
        args.img_dir,
        "image_encoder.pt",
        "image_features.pt"
    )
