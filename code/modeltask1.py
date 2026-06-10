# modeltask1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn import LayerNorm
import os
import re
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
from collections import defaultdict

class GNN1(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN1, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset1"], dict1
        self.args = args

        self.drug_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset1"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset1"], embedding_dim=args.embedding_num)

        self.W1 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))

        self.Linear1 = nn.Sequential(
            nn.Linear(args.embedding_num * 2, args.embedding_num),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num)
        )
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.register_buffer('drug_name_tensor', torch.tensor(drug_name, dtype=torch.long))

    def forward(self, idx):
        kg, args = self.kg, self.args

        adj_tail, adj_relation = self.arrge(kg, self.dict1, args.neighbor_sample_size)

        device = self.drug_name_tensor.device
        adj_tail = torch.tensor(adj_tail, dtype=torch.long, device=device)
        adj_relation = torch.tensor(adj_relation, dtype=torch.long, device=device)

        drug_embedding = self.drug_embed(self.drug_name_tensor)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)

        drug_rel = drug_embedding.reshape((self.drug_name_tensor.size(0), 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((self.drug_name_tensor.size(0), 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat([
            weighted_ent.reshape(self.drug_name_tensor.size(0), args.embedding_num),
            drug_embedding.reshape((self.drug_name_tensor.size(0), args.embedding_num))
        ], dim=1)
        drug_f = self.Linear1(drug_e)

        return drug_f, idx

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=572):
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            if n_neighbor == 0:
                adj_tail[drug_name_id[i]] = np.zeros(neighbor_sample_size, dtype=np.int64)
                adj_relation[drug_name_id[i]] = np.zeros(neighbor_sample_size, dtype=np.int64)
                continue
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[ii][0] for ii in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[ii][1] for ii in sample_indices])
        return adj_tail, adj_relation


class GNN2(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN2, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset2"], dict1
        self.args = args
        self.drug_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset2"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset2"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))

        self.Linear1 = nn.Sequential(
            nn.Linear(args.embedding_num * 2, args.embedding_num),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num)
        )
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.register_buffer('drug_name_tensor', torch.tensor(drug_name, dtype=torch.long))

    def forward(self, arguments):
        kg, args = self.kg, self.args
        gnn1_embedding, idx = arguments

        adj_tail, adj_relation = self.arrge(kg, self.dict1, args.neighbor_sample_size)

        device = self.drug_name_tensor.device
        adj_tail = torch.tensor(adj_tail, dtype=torch.long, device=device)
        adj_relation = torch.tensor(adj_relation, dtype=torch.long, device=device)

        drug_embedding = self.drug_embed(self.drug_name_tensor)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)

        drug_rel = drug_embedding.reshape((self.drug_name_tensor.size(0), 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((self.drug_name_tensor.size(0), 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat([
            weighted_ent.reshape(self.drug_name_tensor.size(0), args.embedding_num),
            drug_embedding.reshape((self.drug_name_tensor.size(0), args.embedding_num))
        ], dim=1)
        drug_f = self.Linear1(drug_e)

        return drug_f, gnn1_embedding, idx

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=572):
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            if n_neighbor == 0:
                adj_tail[drug_name_id[i]] = np.zeros(neighbor_sample_size, dtype=np.int64)
                adj_relation[drug_name_id[i]] = np.zeros(neighbor_sample_size, dtype=np.int64)
                continue
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[ii][0] for ii in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[ii][1] for ii in sample_indices])
        return adj_tail, adj_relation


class GNN3(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN3, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset3"], dict1
        self.args = args
        self.drug_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=67, embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.Linear1 = nn.Sequential(
            nn.Linear(args.embedding_num * 2, args.embedding_num),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num)
        )
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.register_buffer('drug_name_tensor', torch.tensor(drug_name, dtype=torch.long))

    def forward(self, arguments):
        kg, args = self.kg, self.args
        gnn2_embedding, gnn1_embedding, idx = arguments

        adj_tail, adj_relation = self.arrge(kg, self.dict1, args.neighbor_sample_size)

        device = self.drug_name_tensor.device
        adj_tail = torch.tensor(adj_tail, dtype=torch.long, device=device)
        adj_relation = torch.tensor(adj_relation, dtype=torch.long, device=device)

        drug_embedding = self.drug_embed(self.drug_name_tensor)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        drug_rel = drug_embedding.reshape((self.drug_name_tensor.size(0), 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((self.drug_name_tensor.size(0), 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat([
            weighted_ent.reshape(self.drug_name_tensor.size(0), args.embedding_num),
            drug_embedding.reshape((self.drug_name_tensor.size(0), args.embedding_num))
        ], dim=1)
        drug_f = self.Linear1(drug_e)

        return drug_f, gnn2_embedding, gnn1_embedding, idx

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=572, tails_num=570, relations_num=65):
        drug_number = []
        drug_list = []
        for i in drug_name_id:
            drug_number.append(drug_name_id[i])
        for key in kg:
            drug_list.append(key)
        surplus = set(drug_number).difference(set(drug_list))
        for i in list(surplus):
            kg[i].append((tails_num + 1, relations_num + 1))
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            if n_neighbor == 0:
                adj_tail[drug_name_id[i]] = np.zeros(neighbor_sample_size, dtype=np.int64)
                adj_relation[drug_name_id[i]] = np.zeros(neighbor_sample_size, dtype=np.int64)
                continue
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[ii][0] for ii in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[ii][1] for ii in sample_indices])
        return adj_tail, adj_relation

class GNN4(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN4, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset4"], dict1
        self.args = args
        self.drug_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset4"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset4"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))

        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.register_buffer('drug_name_tensor', torch.tensor(drug_name, dtype=torch.long))

    def forward(self, arguments):
        kg, args = self.kg, self.args
        gnn3_embedding, gnn2_embedding, gnn1_embedding, idx = arguments

        adj_tail, adj_relation = self.arrge(kg, self.dict1, args.neighbor_sample_size)

        device = self.drug_name_tensor.device
        adj_tail = torch.tensor(adj_tail, dtype=torch.long, device=device)
        adj_relation = torch.tensor(adj_relation, dtype=torch.long, device=device)

        drug_embedding = self.drug_embed(self.drug_name_tensor)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        drug_rel = drug_embedding.reshape((self.drug_name_tensor.size(0), 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((self.drug_name_tensor.size(0), 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat([
            weighted_ent.reshape(self.drug_name_tensor.size(0), args.embedding_num),
            drug_embedding.reshape((self.drug_name_tensor.size(0), args.embedding_num))
        ], dim=1)
        drug_f = self.Linear1(drug_e)

        return drug_f, gnn3_embedding, gnn2_embedding, gnn1_embedding, idx

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=572):
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            if n_neighbor == 0:
                adj_tail[drug_name_id[i]] = np.zeros(neighbor_sample_size, dtype=np.int64)
                adj_relation[drug_name_id[i]] = np.zeros(neighbor_sample_size, dtype=np.int64)
                continue
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[ii][0] for ii in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[ii][1] for ii in sample_indices])
        return adj_tail, adj_relation

class ImageEncoder(nn.Module):
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


class DynamicModalAttention(nn.Module):
    def __init__(self, n_modalities=5, feat_dim=512, hidden_dim=256):
        super().__init__()
        self.n_modalities = n_modalities
        self.feat_dim = feat_dim

        self.context_encoder = nn.Sequential(
            nn.Linear(feat_dim * n_modalities, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_modalities),
            nn.Softmax(dim=-1)
        )

        self.modality_prior = nn.Parameter(torch.ones(1, n_modalities) / n_modalities)

    def forward(self, modal_features):
        batch_size = modal_features[0].size(0)
        concat_feat = torch.cat(modal_features, dim=-1)
        context = self.context_encoder(concat_feat)
        dynamic_weights = self.weight_generator(context)
        final_weights = dynamic_weights * self.modality_prior
        final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-8)

        weighted_features = []
        for i, feat in enumerate(modal_features):
            weight = final_weights[:, i].view(-1, 1)
            weighted_features.append(feat * weight)

        combined = torch.cat(weighted_features, dim=-1)

        return combined, final_weights

class CrossModalInteraction(nn.Module):

    def __init__(self, feat_dim=512, n_modalities=5):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_modalities = n_modalities

        self.interaction_gates = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feat_dim * 2, feat_dim),
                    nn.Sigmoid()
                ) for _ in range(n_modalities)
            ]) for _ in range(n_modalities)
        ])

        self.balance_lambda = nn.Parameter(torch.ones(n_modalities) * 0.5)

    def forward(self, modal_features):
        enhanced = []

        for i, feat_i in enumerate(modal_features):
            cross_info_list = []
            for j, feat_j in enumerate(modal_features):
                if i == j:
                    continue
                gate = self.interaction_gates[i][j](
                    torch.cat([feat_i, feat_j], dim=-1)
                )
                cross_info_list.append(gate * feat_j)

            if cross_info_list:
                cross_info = torch.stack(cross_info_list).mean(dim=0)
                lam = torch.sigmoid(self.balance_lambda[i])
                enhanced_i = lam * feat_i + (1 - lam) * cross_info
            else:
                enhanced_i = feat_i

            enhanced.append(enhanced_i)

        return enhanced


class FusionLayerWithCSE(nn.Module):
    def __init__(self, args, drug_id_to_name=None):
        super(FusionLayerWithCSE, self).__init__()

        self.drug_id_to_name = drug_id_to_name
        self.proj_dim = 512
        self.single_dim = 256
        self.n_modalities = 5

        drug_pair_features_df = pd.read_parquet('se/sembert.parquet')
        self.drug_pair_features = {}
        for _, row in drug_pair_features_df.iterrows():
            self.drug_pair_features[(row['drug1'], row['drug2'])] = torch.from_numpy(row['feature']).float()

        self.cse_features = torch.load('cse/cse.pt')

        desc_features_df = pd.read_parquet("des/des100.parquet")
        self.desc_features = {row["drug_name"]: torch.tensor(row["feature_vector"]).float()
                              for _, row in desc_features_df.iterrows()}

        def make_single_encoder(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, self.single_dim),
                nn.ReLU(),
                LayerNorm(self.single_dim),
                nn.Dropout(args.dropout)
            )

        self.enc_gnn = make_single_encoder(args.embedding_num * 4)
        self.enc_cse = make_single_encoder(512)
        self.enc_desc = make_single_encoder(768)
        self.enc_img = make_single_encoder(256)

        def make_pair_proj():
            return nn.Sequential(
                nn.Linear(self.single_dim * 4, self.proj_dim),
                nn.ReLU(),
                LayerNorm(self.proj_dim),
                nn.Dropout(args.dropout)
            )

        self.proj_gnn = make_pair_proj()
        self.proj_cse = make_pair_proj()
        self.proj_desc = make_pair_proj()
        self.proj_img = make_pair_proj()

        self.proj_bert = nn.Sequential(
            nn.Linear(768, self.proj_dim),
            nn.ReLU(),
            LayerNorm(self.proj_dim),
            nn.Dropout(args.dropout)
        )

        self.missing_bert = nn.Parameter(torch.randn(1, 768) * 0.02)
        self.missing_cse = nn.Parameter(torch.randn(1, 512) * 0.02)
        self.missing_desc = nn.Parameter(torch.randn(1, 768) * 0.02)
        self.missing_img = nn.Parameter(torch.randn(1, 256) * 0.02)

        self.image_encoder = ImageEncoder(embed_dim=256)
        image_encoder_path = "img/image_encoder.pt"
        if os.path.exists(image_encoder_path):
            self.image_encoder.load_state_dict(torch.load(image_encoder_path, map_location='cpu'))
            print("✓ Successfully loaded pretrained image_encoder.pt")
        else:
            print(f"⚠ Warning: {image_encoder_path} not found, using random initialization")

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.img_dir = "img/01drug_3d_png"
        self.drug_to_images = self._build_drug_to_images()
        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.dynamic_attention = DynamicModalAttention(
            n_modalities=self.n_modalities,
            feat_dim=self.proj_dim,
            hidden_dim=256
        )

        self.cross_modal_interaction = CrossModalInteraction(
            feat_dim=self.proj_dim,
            n_modalities=self.n_modalities
        )

        self.meta_gate = nn.Sequential(
            nn.Linear(self.proj_dim * self.n_modalities, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(self.proj_dim * self.n_modalities, args.embedding_num * 4),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 4),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 4, args.embedding_num * 2),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 2),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 2, 65)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _build_drug_to_images(self):
        mapping = defaultdict(list)
        if not os.path.exists(self.img_dir):
            print(f"Warning: Image directory {self.img_dir} does not exist!")
            return mapping
        for file in os.listdir(self.img_dir):
            if not file.endswith(".png"):
                continue
            m = re.match(r"(DB\d+)_view[1-4]\.png", file)
            if m:
                drug_id = m.group(1)
                mapping[drug_id].append(os.path.join(self.img_dir, file))
        print(f"Loaded {len(mapping)} drugs with image files.")
        return mapping

    def _get_image_feature(self, drug_key, device):
        paths = self.drug_to_images.get(drug_key, [])
        if not paths:
            return self.missing_img.to(device).view(1, -1)
        feats = []
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = self.img_transform(img).unsqueeze(0).to(device)
                with torch.set_grad_enabled(self.training):
                    feat = self.image_encoder(img_tensor)
                feats.append(feat.squeeze(0))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                continue
        if not feats:
            return self.missing_img.to(device).view(1, -1)
        return torch.stack(feats).mean(dim=0).view(1, -1)

    def get_feat(self, dic, name, missing, device):
        f = dic.get(name)
        if f is None:
            return missing.to(device)
        if not isinstance(f, torch.Tensor):
            f = torch.tensor(f, dtype=torch.float)
        return f.to(device).view(1, -1)

    def pair_fusion(self, A, B):
        return torch.cat([A, B, torch.abs(A - B), A * B], dim=1)

    def forward(self, arguments):
        gnn4_embedding, gnn3_embedding, gnn2_embedding, gnn1_embedding, idx = arguments

        if isinstance(idx, torch.Tensor):
            idx_list = idx.detach().cpu().numpy().tolist()
        else:
            idx_list = idx

        drugA = [i[0] for i in idx_list]
        drugB = [i[1] for i in idx_list]

        drugA_names = [self.drug_id_to_name[i] for i in drugA]
        drugB_names = [self.drug_id_to_name[i] for i in drugB]

        device = next(self.parameters()).device

        gnnA = torch.cat([gnn1_embedding[drugA], gnn2_embedding[drugA],
                          gnn3_embedding[drugA], gnn4_embedding[drugA]], dim=1).float().to(device)
        gnnB = torch.cat([gnn1_embedding[drugB], gnn2_embedding[drugB],
                          gnn3_embedding[drugB], gnn4_embedding[drugB]], dim=1).float().to(device)
        gnnA = self.enc_gnn(gnnA)
        gnnB = self.enc_gnn(gnnB)
        gnn_feat = self.proj_gnn(self.pair_fusion(gnnA, gnnB))

        bert_list = []
        for a, b in zip(drugA_names, drugB_names):
            f = self.drug_pair_features.get((a, b))
            if f is None:
                bert_list.append(self.missing_bert.to(device))
            else:
                bert_list.append(f.to(device).view(1, -1))
        bert_feats = torch.cat(bert_list, dim=0)
        bert_feat = self.proj_bert(bert_feats)

        cseA = torch.cat([self.get_feat(self.cse_features, n, self.missing_cse, device) for n in drugA_names])
        cseB = torch.cat([self.get_feat(self.cse_features, n, self.missing_cse, device) for n in drugB_names])
        cseA = self.enc_cse(cseA)
        cseB = self.enc_cse(cseB)
        cse_feat = self.proj_cse(self.pair_fusion(cseA, cseB))

        descA = torch.cat([self.get_feat(self.desc_features, n, self.missing_desc, device) for n in drugA_names])
        descB = torch.cat([self.get_feat(self.desc_features, n, self.missing_desc, device) for n in drugB_names])
        descA = self.enc_desc(descA)
        descB = self.enc_desc(descB)
        desc_feat = self.proj_desc(self.pair_fusion(descA, descB))

        imgA_list, imgB_list = [], []
        for nameA, nameB in zip(drugA_names, drugB_names):
            imgA_list.append(self._get_image_feature(nameA, device))
            imgB_list.append(self._get_image_feature(nameB, device))
        imgA = torch.cat(imgA_list, dim=0)
        imgB = torch.cat(imgB_list, dim=0)
        imgA = self.enc_img(imgA)
        imgB = self.enc_img(imgB)
        img_feat = self.proj_img(self.pair_fusion(imgA, imgB))

        modal_features = [gnn_feat, bert_feat, cse_feat, desc_feat, img_feat]

        enhanced_features = self.cross_modal_interaction(modal_features)

        combined_dynamic, attention_weights = self.dynamic_attention(enhanced_features)

        simple_concat = torch.cat(enhanced_features, dim=-1)
        gate = self.meta_gate(combined_dynamic)
        final_features = gate * combined_dynamic + (1 - gate) * simple_concat

        output = self.fullConnectionLayer(final_features)

        return output, attention_weights