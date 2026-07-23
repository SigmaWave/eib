from __future__ import annotations
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLP(nn.Module):
    def __init__(self, in_ch: int, hid: int = 256, heads1: int = 4, dropout: float = 0.6,
                 num_classes: int = 0, num_global_nodes: int = 0,
                 num_rels: int = 1, num_cats: int = 1, use_edge_types: bool = True):
        super().__init__()

        self.use_edge_types = use_edge_types
        self.type_emb = nn.Embedding(num_classes + 1, hid, padding_idx=0)
        nn.init.xavier_uniform_(self.type_emb.weight)

        rel_dim = 16 if use_edge_types else 0
        cat_dim = 16 if use_edge_types else 0
        if use_edge_types:
            self.rel_emb = nn.Embedding(num_rels + 1, rel_dim, padding_idx=0)
            self.cat_emb = nn.Embedding(num_cats + 1, cat_dim, padding_idx=0)
            nn.init.xavier_uniform_(self.rel_emb.weight)
            nn.init.xavier_uniform_(self.cat_emb.weight)

        self.edge_dim_internal = rel_dim + cat_dim + 1

        self.node_emb = nn.Embedding(num_global_nodes, hid, padding_idx=0)
        nn.init.xavier_uniform_(self.node_emb.weight)

        self.lin_input = nn.Linear(in_ch, hid)
        self.ln1 = nn.LayerNorm(hid)
        self.gat1 = GATConv(hid, hid, heads=heads1, dropout=dropout, edge_dim=self.edge_dim_internal)
        self.ln2 = nn.LayerNorm(hid * heads1)
        self.gat2 = GATConv(hid * heads1, hid, heads=1, concat=False, dropout=dropout, edge_dim=self.edge_dim_internal)

        self.classifier = nn.Linear(hid, num_classes) if num_classes > 0 else None
        self.elu = nn.ELU()
        self.drop = nn.Dropout(dropout)
        self.id_dropout_prob = 0.5

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, n_id = data.x, data.edge_index, data.n_id
        edge_attr_packed = getattr(data, "edge_attr", None)

        node_type_ids = getattr(data, "node_type", None)
        if node_type_ids is None:
            node_type_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if edge_attr_packed is not None and edge_attr_packed.shape[0] > 0:
            r_ids = edge_attr_packed[:, 0].long()
            c_ids = edge_attr_packed[:, 1].long()
            w_val = edge_attr_packed[:, 2].unsqueeze(1)
            if self.use_edge_types:
                r_vec = self.rel_emb(r_ids)
                c_vec = self.cat_emb(c_ids)
                edge_attr = torch.cat([r_vec, c_vec, w_val], dim=1)
            else:
                edge_attr = w_val
        else:
            edge_attr = torch.empty((0, self.edge_dim_internal), device=x.device)

        x_dense = self.lin_input(x)
        x_type = self.type_emb(node_type_ids)

        if self.training:
            mask = torch.rand(n_id.size(0), device=n_id.device) > self.id_dropout_prob
            id_emb = self.node_emb(n_id) * mask.unsqueeze(1)
        else:
            id_emb = self.node_emb(n_id)

        x = x_dense + x_type + id_emb
        x = self.ln1(x)
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = self.ln2(x)
        x = self.elu(x)
        x = self.drop(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        return x

    def classify(self, emb: torch.Tensor) -> torch.Tensor:
        if self.classifier is None: raise ValueError("No classifier head.")
        return self.classifier(emb)
