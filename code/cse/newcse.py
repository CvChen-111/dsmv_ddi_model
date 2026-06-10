# enhanced_cse_extractor.py
"""
增强版 CSE 通道特征提取脚本
- Graph-CSE: GINConv + GCNConv + TransformerConv + Attention Pooling
- 可选融合 ChemBERTa embedding
- 输出单药物 embedding 和药物对级别特征 [A, B, A*B, |A-B|]
Usage:
    python enhanced_cse_extractor.py \
        --smiles_csv 572drug_smile.csv \
        --output_drug_pt drug_graph_features.pt \
        --output_pair_pt drug_pair_cse_features.pt \
        --use_chemberta chemberta_features.pt \
        --device cuda
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import (
    GINConv,
    GCNConv,
    TransformerConv,
    GlobalAttention,
    global_mean_pool,
    global_add_pool,
)
from torch_geometric.data import Batch

# ---- Utility: your existing preprocessing function should produce node/edge features ----
# You mentioned earlier a helper: get_mol_edge_list_and_feat_mtx(mol)
# Replace the import below with the correct path to your data_preprocessing function.
try:
    from data_preprocessing_t import get_mol_edge_list_and_feat_mtx
except Exception:
    # fallback stub (you must implement/replace with your actual function)
    def get_mol_edge_list_and_feat_mtx(mol):
        """
        Should return: edge_index (torch.LongTensor, [2, E]),
                       node_feat (torch.FloatTensor, [N, node_feat_dim]),
                       edge_feat (torch.FloatTensor, [E, edge_feat_dim])
        Implemented by user's preprocessing module.
        """
        raise RuntimeError("Please provide real get_mol_edge_list_and_feat_mtx in data_preprocessing_t.py")

# ---------------- Model ----------------
class GraphCSE(nn.Module):
    def __init__(
        self,
        in_node_dim,
        in_edge_dim,
        hidden_dim=256,
        output_dim=256,
        gnn_dropout=0.1,
        use_residual=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_residual = use_residual

        # initial encoders
        self.node_encoder = nn.Linear(in_node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(in_edge_dim, hidden_dim)

        # normalization
        self.node_norm1 = nn.LayerNorm(hidden_dim)
        self.node_norm2 = nn.LayerNorm(hidden_dim)

        # GINConv block
        nn1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gin1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gin2 = GINConv(nn2)

        # GCN Conv to add complementary signal
        self.gcn = GCNConv(hidden_dim, hidden_dim)

        # TransformerConv (with edge_attr) - heads set so concat output = hidden_dim
        heads = 4
        out_per_head = hidden_dim // heads
        assert out_per_head * heads == hidden_dim, "hidden_dim must be divisible by heads"
        self.transf = TransformerConv(hidden_dim, out_per_head, heads=heads, edge_dim=hidden_dim, dropout=gnn_dropout)

        # Pooling: GlobalAttention (learnable gate)
        self.att_gate_nn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.pool = GlobalAttention(self.att_gate_nn)

        # final projection
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        # dropout
        self.dropout = nn.Dropout(gnn_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Data):
        """
        data: torch_geometric.data.Data or Batch with attributes:
            x: [num_nodes, in_node_dim]
            edge_index: [2, E]
            edge_attr: [E, in_edge_dim]
            batch: [num_nodes]
        returns: graph embedding [num_graphs, output_dim]
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # encode
        x0 = self.node_encoder(x)  # [N, H]
        e0 = self.edge_encoder(edge_attr) if edge_attr is not None else None
        x0 = F.relu(self.node_norm1(x0))

        # GIN blocks with residual
        h = self.gin1(x0, edge_index)
        h = F.relu(self.node_norm2(h))
        if self.use_residual:
            h = h + x0
        h = self.dropout(h)

        h2 = self.gin2(h, edge_index)
        h2 = F.relu(h2)
        if self.use_residual:
            h2 = h2 + h
        h2 = self.dropout(h2)

        # additional GCN
        h3 = self.gcn(h2, edge_index)
        h3 = F.relu(h3)
        if self.use_residual:
            h3 = h3 + h2
        h3 = self.dropout(h3)

        # transformer conv (uses encoded edge_attr)
        if e0 is not None:
            h4 = self.transf(h3, edge_index, e0)
        else:
            # pass zeros if edge_attr missing
            _e = torch.zeros(edge_index.shape[1], self.hidden_dim, device=h3.device)
            h4 = self.transf(h3, edge_index, _e)
        h4 = F.relu(h4)
        if h4.dim() == 3:  # sometimes TransformerConv returns [N, heads, out_per_head]
            h4 = h4.mean(dim=1)
        h4 = self.dropout(h4)

        # pooling (Global Attention)
        g = self.pool(h4, batch)  # [num_graphs, hidden_dim]

        # project
        out = self.project(g)  # [num_graphs, output_dim]
        return out


# ------------- Helpers -------------
def build_mol_graph(smiles: str, name: str):
    """Construct torch_geometric.data.Data from SMILES using user's preprocessing function."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[Warning] cannot parse SMILES for {name}")
        return None
    try:
        edge_index, node_feat, edge_feat = get_mol_edge_list_and_feat_mtx(mol)
        # ensure tensors
        if not isinstance(node_feat, torch.Tensor):
            node_feat = torch.tensor(node_feat, dtype=torch.float)
        if not isinstance(edge_feat, torch.Tensor):
            edge_feat = torch.tensor(edge_feat, dtype=torch.float)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_feat, batch=torch.zeros(node_feat.shape[0], dtype=torch.long))
        return data
    except Exception as e:
        print(f"[Error] build graph failed for {name}: {e}")
        return None


def stack_graphs_to_batch(data_list):
    """Convert dict(name->Data) -> Batch for model forward in batches."""
    from torch_geometric.data import Batch
    return Batch.from_data_list(list(data_list.values()))


# ------------- Feature fusion and pair creation -------------
class CSEFusionHead(nn.Module):
    """
    Given single-drug embeddings (graph_emb dim G, optional chemberta dim C),
    produce fused single-drug embedding and pair-level feature projector that forms:
      pair = [A, B, A*B, |A-B|] -> projected to pair_dim
    """
    def __init__(self, graph_dim=256, chem_dim=0, fused_dim=512, pair_proj_dim=512):
        super().__init__()
        self.graph_dim = graph_dim
        self.chem_dim = chem_dim
        self.in_dim = graph_dim + chem_dim
        self.fused_dim = fused_dim

        self.fuse_proj = nn.Sequential(
            nn.Linear(self.in_dim, fused_dim),
            nn.ReLU(),
            nn.LayerNorm(fused_dim)
        )
        # pair projection: input = fused_dim * 4 (A,B,A*B,|A-B|)
        self.pair_proj = nn.Sequential(
            nn.Linear(fused_dim * 4, pair_proj_dim),
            nn.ReLU(),
            nn.LayerNorm(pair_proj_dim)
        )

    def forward_single(self, emb_graph, emb_chem=None):
        if emb_chem is not None:
            emb = torch.cat([emb_graph, emb_chem], dim=-1)
        else:
            emb = emb_graph
        return self.fuse_proj(emb)  # [batch, fused_dim]

    def forward_pair(self, embA, embB):
        # embA, embB: [batch, fused_dim] or [fused_dim]
        prod = embA * embB
        diff = torch.abs(embA - embB)
        concat = torch.cat([embA, embB, prod, diff], dim=-1)
        return self.pair_proj(concat)  # [batch, pair_proj_dim]


# ------------- Main pipeline -------------
def extract_graph_features(
    smiles_csv,
    output_drug_pt,
    output_pair_pt,
    model_device="cuda",
    graph_hidden_dim=256,
    graph_output_dim=256,
    fuse_chembert_pt=None,
    fused_output_dim=512,
    pair_proj_dim=512,
    batch_size=64
):
    # 1) read smiles file
    df = pd.read_csv(smiles_csv)
    drug_names = df["name"].tolist()
    smiles_list = df["smiles"].tolist()

    # 2) build mol graphs
    mol_dict = {}
    for name, smi in zip(drug_names, smiles_list):
        data = build_mol_graph(smi, name)
        if data is not None:
            mol_dict[name] = data
    if len(mol_dict) == 0:
        raise RuntimeError("No valid molecule graphs constructed. Check SMILES and preprocessing.")

    # device
    device = torch.device(model_device if torch.cuda.is_available() and model_device == "cuda" else "cpu")

    # 3) init model
    sample = next(iter(mol_dict.values()))
    node_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1]
    gmodel = GraphCSE(in_node_dim=node_dim, in_edge_dim=edge_dim, hidden_dim=graph_hidden_dim, output_dim=graph_output_dim)
    gmodel.to(device)
    gmodel.eval()

    # 4) optionally load chemberta single-drug features
    chem_features = None
    if fuse_chembert_pt:
        if os.path.exists(fuse_chembert_pt):
            chem_features = torch.load(fuse_chembert_pt, map_location="cpu")
            # ensure tensors on CPU then moved later
            print(f"[Info] Loaded ChemBERTa features for {len(chem_features)} drugs.")
        else:
            raise FileNotFoundError(f"chemberta features file not found: {fuse_chembert_pt}")

    # 5) batch forward to compute graph embeddings
    drug_graph_emb = {}
    names = list(mol_dict.keys())
    # we will forward in small batches by creating Batch objects
    from torch_geometric.data import Batch
    B = []
    idx_to_name = []
    for name in names:
        B.append(mol_dict[name])
        idx_to_name.append(name)
        # flush per batch_size
        if len(B) >= batch_size:
            batch = Batch.from_data_list(B).to(device)
            with torch.no_grad():
                out = gmodel(batch)  # [batch_size, graph_output_dim]
            out = out.cpu()
            for i, nm in enumerate(idx_to_name[:len(B)]):
                drug_graph_emb[nm] = out[i]
            B = []
            idx_to_name = []
    # remainder
    if len(B) > 0:
        batch = Batch.from_data_list(B).to(device)
        with torch.no_grad():
            out = gmodel(batch)
        out = out.cpu()
        for i, nm in enumerate(idx_to_name[:len(B)]):
            drug_graph_emb[nm] = out[i]

    print(f"[Info] Extracted graph embeddings for {len(drug_graph_emb)} drugs.")

    # 6) optionally fuse chemberta: prepare chem tensors aligned to names (or None)
    chem_tensors_aligned = {}
    if chem_features is not None:
        # chem_features: dict name->tensor (torch)
        for name in drug_graph_emb.keys():
            if name in chem_features:
                t = chem_features[name]
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t, dtype=torch.float)
                chem_tensors_aligned[name] = t
            else:
                chem_tensors_aligned[name] = None  # missing

    # 7) create fusion head and produce fused single-drug embeddings
    chem_dim = 0
    if chem_features is not None:
        # infer chem dim from first available
        first = next((chem_features[k] for k in chem_features if chem_features[k] is not None), None)
        if first is None:
            chem_dim = 0
        else:
            chem_dim = first.shape[0] if isinstance(first, torch.Tensor) else len(first)

    fusion_head = CSEFusionHead(graph_dim=graph_output_dim, chem_dim=chem_dim, fused_dim=fused_output_dim, pair_proj_dim=pair_proj_dim)
    fusion_head.to(device)
    fusion_head.eval()

    drug_fused_emb = {}
    for name, gemb in drug_graph_emb.items():
        gemb_t = gemb.unsqueeze(0).to(device)  # [1, G]
        chem = None
        if chem_features is not None and chem_tensors_aligned.get(name) is not None:
            chem = chem_tensors_aligned[name].unsqueeze(0).to(device)
        with torch.no_grad():
            fused = fusion_head.forward_single(gemb_t, chem)  # [1, fused_dim]
        drug_fused_emb[name] = fused.squeeze(0).cpu()

    print(f"[Info] Produced fused single-drug embeddings for {len(drug_fused_emb)} drugs.")

    # 8) save single-drug features (graph-only and fused if exists)
    torch.save(drug_graph_emb, output_drug_pt.replace(".pt", ".graph.pt"))
    torch.save(drug_fused_emb, output_drug_pt)  # fused if chem provided else same as graph (projected)
    print(f"[Done] Saved graph features -> {output_drug_pt.replace('.pt', '.graph.pt')} and fused -> {output_drug_pt}")

    # 9) build pair-level features [A,B,A*B,|A-B|] and project
    pair_dict = {}
    names_sorted = sorted(list(drug_fused_emb.keys()))
    # To limit size, you might only compute for pairs in dataset; here we compute all pairs (N^2). Careful with memory.
    for i, a in enumerate(tqdm(names_sorted, desc="Building pair features")):
        embA = drug_fused_emb[a].unsqueeze(0).to(device)  # [1, F]
        for b in names_sorted:  # you can restrict to b >= a to avoid duplication
            embB = drug_fused_emb[b].unsqueeze(0).to(device)
            with torch.no_grad():
                pair_proj = fusion_head.forward_pair(embA, embB)  # [1, pair_proj_dim]
            pair_dict[(a, b)] = pair_proj.squeeze(0).cpu()
        # optional: free GPU memory per row
        torch.cuda.empty_cache()

    torch.save(pair_dict, output_pair_pt)
    print(f"[Done] Saved pair features -> {output_pair_pt} (total pairs: {len(pair_dict)})")

    return {
        "single_graph_pt": output_drug_pt.replace(".pt", ".graph.pt"),
        "single_fused_pt": output_drug_pt,
        "pair_pt": output_pair_pt
    }


# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_csv", type=str, required=True, help="CSV with columns: name, smiles")
    parser.add_argument("--output_drug_pt", type=str, default="drug_fused_features.pt")
    parser.add_argument("--output_pair_pt", type=str, default="drug_pair_cse_features.pt")
    parser.add_argument("--use_chemberta", type=str, default=None, help="optional path to chemberta single-drug pt dict")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--graph_hidden_dim", type=int, default=256)
    parser.add_argument("--graph_output_dim", type=int, default=256)
    parser.add_argument("--fused_output_dim", type=int, default=512)
    parser.add_argument("--pair_proj_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    extract_graph_features(
        smiles_csv=args.smiles_csv,
        output_drug_pt=args.output_drug_pt,
        output_pair_pt=args.output_pair_pt,
        model_device=args.device,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_output_dim=args.graph_output_dim,
        fuse_chembert_pt=args.use_chemberta,
        fused_output_dim=args.fused_output_dim,
        pair_proj_dim=args.pair_proj_dim,
        batch_size=args.batch_size
    )
