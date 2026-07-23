from __future__ import annotations
from torch_geometric.data import Data
from typing import Dict, List, Tuple
import hashlib
import math
import networkx as nx
import numpy as np
import pandas as pd
import random
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import re
import torch

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def safe_lap_eigs(G: nx.Graph, k: int = 3) -> np.ndarray:
    try:
        if G.number_of_nodes() == 0: return np.zeros((0, k), dtype=float)

        UG = G.to_undirected()
        A = nx.to_scipy_sparse_array(UG, dtype=float, format="csr")
        d = np.asarray(A.sum(1)).ravel()
        D = sp.diags(d)
        L = D - A
        n = L.shape[0]
        if n <= 2 or k <= 0: return np.zeros((n, k), dtype=float)
        kk = min(k, n - 2)
        if kk <= 0: return np.zeros((n, k), dtype=float)

        vals, vecs = spla.eigsh(L.asfptype(), k=kk, which='SM', tol=1e-2)
        X = vecs.real
        if X.shape[1] < k:
            pad = np.zeros((X.shape[0], k - X.shape[1]))
            X = np.hstack([X, pad])
        elif X.shape[1] > k:
            X = X[:, :k]
        return X
    except Exception:
        return np.zeros((G.number_of_nodes(), k), dtype=float)

def sanitize_filename(name: str) -> str:
    name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
    return re.sub(r'(?u)[^-\w.]', '', name)

# FEATURE CONSTRUCTION
def get_name_hash_vector(name: str, dim: int = 16) -> np.ndarray:
    hash_digest = hashlib.md5(name.encode('utf-8')).hexdigest()
    seed_int = int(hash_digest, 16) % (2**32)
    rng = np.random.RandomState(seed_int)
    return rng.normal(0, 1, size=(dim,))

def build_node_features(G: nx.Graph, node_type_map: Dict, type2id: Dict, dim_hash: int = 32) -> Tuple[np.ndarray, List, List[int]]:
    nodes = list(G.nodes())
    types = [node_type_map.get(u, "UNK") for u in nodes]
    type_ids = [type2id.get(t, 0) for t in types]

    deg = np.array([G.degree(u) for u in nodes], dtype=float)
    deg_norm = np.log1p(deg).reshape(-1, 1)

    name_feats = np.array([get_name_hash_vector(str(u), dim=dim_hash) for u in nodes])

    lap = safe_lap_eigs(G, k=3)
    if lap.shape[0] != len(nodes): lap = np.zeros((len(nodes), 3), dtype=float)

    X = np.hstack([name_feats, deg_norm, lap])
    return X, nodes, type_ids

def build_rel_map(df: pd.DataFrame) -> Dict[str, int]:
    rels = sorted(df['rel'].dropna().unique())
    return {r: i for i, r in enumerate(rels)}

def build_rel_category_map(df: pd.DataFrame) -> Dict[str, int]:
    cats = set(df['rel_category'].dropna().unique())
    cats.add("UNK")
    return {c: i for i, c in enumerate(sorted(list(cats)))}

def build_type_map(df: pd.DataFrame) -> Dict[str, int]:
    types = set(df['sub_type'].dropna().unique()) | set(df['obj_type'].dropna().unique())
    types.add("UNK")
    return {t: i for i, t in enumerate(sorted(list(types)))}

def pack_pyg_data(G: nx.Graph,
                  X: np.ndarray,
                  nodes: List,
                  type_ids_list: List[int],
                  type2id: Dict[str, int],
                  node_type_map: Dict[str, str],
                  num_rels: int,
                  num_cats: int,
                  use_edge_types: bool = True) -> Tuple[Data, Dict]:

    idx_map = {u: i for i, u in enumerate(nodes)}
    src_idx, dst_idx = [], []
    rel_ids, cat_ids, weights = [], [], []

    for u, v, edata in G.edges(data=True):
        if u not in idx_map or v not in idx_map: continue
        src_idx.append(idx_map[u])
        dst_idx.append(idx_map[v])

        w = float(edata.get("w", 1.0))
        weights.append(math.log1p(w))

        if use_edge_types:
            rel_ids.append(int(edata.get("rel_id", 0)))
            cat_ids.append(int(edata.get("cat_id", 0)))
        else:
            rel_ids.append(0)
            cat_ids.append(0)

    if len(src_idx) > 0:
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        r_vec = torch.tensor(rel_ids, dtype=torch.long)
        c_vec = torch.tensor(cat_ids, dtype=torch.long)
        w_vec = torch.tensor(weights, dtype=torch.float)
        edge_attr = torch.stack([r_vec, c_vec, w_vec], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)

    x = torch.tensor(X, dtype=torch.float)
    node_type_tensor = torch.tensor(type_ids_list, dtype=torch.long)
    y = node_type_tensor.clone()

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.node_type = node_type_tensor
    return data, idx_map
