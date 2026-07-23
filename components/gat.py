from __future__ import annotations
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict
import gc
import networkx as nx
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
import os
import argparse

from components.gat_loss import bpr_link_loss, ce_node_loss, findkg_loss, info_nce_loss
from components.gat_model import GATLP
from utils.data_utils import load_df_from_csv
from utils.gat_utils import build_node_features, build_rel_map, build_rel_category_map, build_type_map, get_device, pack_pyg_data, set_seed, SEED
from utils.logger import Logger

Logger.setup_file("logs")

# CONFIG
FREQ = os.environ.get("FREQ", "M")
NEG_PER_POS = 50
DEBUG_N = int(os.environ.get("DEBUG_N", "0"))
HYBRID_CE_WEIGHT = 0.5
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)
WINDOW_SIZE = 3
PATIENCE = 20
MIN_DELTA = 1e-4

# TRAIN vs LOAD: Set to "false" to skip training and load existing weights if available
TRAIN_MODE = os.environ.get("TRAIN_MODE", "true").lower() == "true"

def sample_neg_edges(G, pos_edges, k=10, allowed_nodes=None):
    if allowed_nodes is None: nodes = list(G.nodes())
    else: nodes = list(allowed_nodes)
    if len(nodes) < 2: return []

    E = set(G.edges()) | {(v, u) for (u, v) in G.edges()}
    all_neg_edges = []

    for (u, v) in pos_edges:
        current_neg = []
        attempts = 0
        while len(current_neg) < k and attempts < k * 3:
            attempts += 1
            b = random.choice(nodes)
            if b != u and (u, b) not in E:
                current_neg.append((u, b))

        if len(current_neg) < k:
             while len(current_neg) < k and len(current_neg) > 0:
                 current_neg.append(random.choice(current_neg))
        all_neg_edges.extend(current_neg)
    return all_neg_edges

def edge_index_from_pairs(pairs, idx_map):
    src = [idx_map[u] for (u, _) in pairs]
    dst = [idx_map[v] for (_, v) in pairs]
    return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)

@torch.no_grad()
def evaluate_lp(emb, test_pos, test_neg, idx_map, train_nodes_set):
    P = len(test_pos)
    if P == 0: return {}

    pos_src, pos_dst = edge_index_from_pairs(test_pos, idx_map)
    neg_src, neg_dst = edge_index_from_pairs(test_neg, idx_map)

    s_pos = (emb[pos_src] * emb[pos_dst]).sum(dim=1).cpu().numpy()
    s_neg_all = (emb[neg_src] * emb[neg_dst]).sum(dim=1).cpu().numpy()

    neg_k = len(test_neg) // P
    if len(s_neg_all) != P * neg_k:
         s_neg_all = s_neg_all[:P * neg_k]
    s_neg = s_neg_all.reshape(P, neg_k)

    ranks_all = []
    ranks_seen = []
    ranks_unseen = []

    for i in range(P):
        u, v = test_pos[i]
        rank = (s_neg[i] >= s_pos[i]).sum() + 1
        ranks_all.append(rank)

        if u in train_nodes_set and v in train_nodes_set:
            ranks_seen.append(rank)
        else:
            ranks_unseen.append(rank)

    def calc_metrics(rank_list):
        if not rank_list: return None
        n = len(rank_list)
        return {
            "MRR": np.mean([1.0/r for r in rank_list]),
            "Hits@1": sum(1 for r in rank_list if r <= 1) / n,
            "Hits@3": sum(1 for r in rank_list if r <= 3) / n,
            "Hits@10": sum(1 for r in rank_list if r <= 10) / n,
            "Count": n
        }

    results = {}
    m_all = calc_metrics(ranks_all)
    if m_all: results.update(m_all)

    m_seen = calc_metrics(ranks_seen)
    if m_seen:
        for k, v in m_seen.items(): results[f"Seen_{k}"] = v
    else:
        results["Seen_Count"] = 0

    m_unseen = calc_metrics(ranks_unseen)
    if m_unseen:
        for k, v in m_unseen.items(): results[f"Unseen_{k}"] = v
    else:
        results["Unseen_Count"] = 0

    return results

@torch.no_grad()
def validate_one_epoch(model, data, G_val, idx_map, loss_type="bpr"):
    model.eval()
    emb = model(data)
    val_loss = torch.tensor(0.0, device=emb.device)
    allowed = set(idx_map.keys())
    val_edges = [(u, v) for u, v in G_val.edges() if u in allowed and v in allowed]

    if loss_type in ["bpr", "hybrid", "findkg", "infonce"] and val_edges:
        neg_edges = sample_neg_edges(G_val, val_edges, k=NEG_PER_POS, allowed_nodes=allowed)
        if neg_edges:
            p_s, p_d = edge_index_from_pairs(val_edges, idx_map)
            n_s, n_d = edge_index_from_pairs(neg_edges, idx_map)
            if loss_type == "findkg":
                val_loss += findkg_loss(emb, p_s, p_d, n_s, n_d)
            elif loss_type == "infonce" or loss_type == "hybrid":
                 val_loss += info_nce_loss(emb, p_s, p_d, n_s, n_d, temperature=0.1)
            else:
                val_loss += bpr_link_loss(emb, p_s, p_d, n_s, n_d, margin=5.0)

    if loss_type in ["ce", "hybrid"]:
        val_nodes = [u for u in G_val.nodes() if u in idx_map]
        if val_nodes:
            val_indices = [idx_map[u] for u in val_nodes]
            val_mask = torch.tensor(val_indices, dtype=torch.long, device=emb.device)
            logits = model.classify(emb)
            target = data.y[val_mask]
            pred = logits[val_mask]
            ce_val = F.cross_entropy(pred, target)
            val_loss += (HYBRID_CE_WEIGHT * ce_val) if loss_type == "hybrid" else ce_val

    return val_loss

def run_experiment_case(df: pd.DataFrame, case_config: Dict,
                        global_type2id: Dict, global_node2id: Dict) -> Dict[str, float]:

    case_name = case_config["name"]
    use_edge_types = case_config["edge_types"]
    loss_type = case_config["loss"]
    epochs = case_config["epochs"]
    learning_rate = case_config["learning_rate"]
    model_fs_name = case_config["fs_name"]

    device = get_device()
    Logger.info(f"\n>>> Running {case_name} [Mode={('TRAIN' if TRAIN_MODE else 'LOAD')}] on Device: {device}")

    set_seed(SEED)

    model_weight_dir = WEIGHTS_DIR / model_fs_name
    model_weight_dir.mkdir(parents=True, exist_ok=True)

    ts = pd.to_datetime(df['date'])
    periods = pd.Index(ts.dt.to_period(FREQ).astype(str).unique()).sort_values()

    rel2id = build_rel_map(df)
    rel_cat2id = build_rel_category_map(df)
    num_rels = len(rel2id) if len(rel2id) > 0 else 1
    num_cats = len(rel_cat2id) if len(rel_cat2id) > 0 else 1

    if len(periods) < 2:
        raise ValueError(f"Only {len(periods)} period found.")

    metrics_acc = {}
    num_global_nodes = len(global_node2id) + 1

    # Feature Dim
    sample_G = build_graph_from_df(df[df['date'].dt.to_period(FREQ).astype(str) == periods[0]], rel2id, rel_cat2id)
    node_map_sample = {u: sample_G.nodes[u].get("node_type", "UNK") for u in sample_G.nodes()}
    X_sample, _, _ = build_node_features(sample_G, node_map_sample, global_type2id)
    in_channels = X_sample.shape[1]

    # Initialize Model and Move to Device
    model = GATLP(
        in_ch=in_channels,
        hid=256,
        num_classes=len(global_type2id),
        num_global_nodes=num_global_nodes,
        dropout=0.3,
        num_rels=num_rels,
        num_cats=num_cats,
        use_edge_types=use_edge_types
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(len(periods) - 1):
        start_idx = max(0, i + 1 - WINDOW_SIZE)
        train_p = periods[start_idx : i+1]
        val_p = periods[i+1]

        weight_path = model_weight_dir / f"{val_p}.pt"

        df_train = df[pd.to_datetime(df['date']).dt.to_period(FREQ).astype(str).isin(train_p)]
        df_val = df[pd.to_datetime(df['date']).dt.to_period(FREQ).astype(str) == val_p]

        G_train = build_graph_from_df(df_train, rel2id, rel_cat2id)
        G_val = build_graph_from_df(df_val, rel2id, rel_cat2id)

        node_map_train = {u: G_train.nodes[u].get("node_type", "UNK") for u in G_train.nodes()}
        X_train, nodes_train, type_ids_train = build_node_features(G_train, node_map_train, global_type2id)
        data_train, idx_map_train = pack_pyg_data(
            G_train, X_train, nodes_train, type_ids_train,
            global_type2id, node_map_train, num_rels, num_cats, use_edge_types
        )
        data_train = data_train.to(device)

        node_map_val = {u: G_val.nodes[u].get("node_type", "UNK") for u in G_val.nodes()}
        X_val, nodes_val, type_ids_val = build_node_features(G_val, node_map_val, global_type2id)
        data_val, idx_map_val = pack_pyg_data(
            G_val, X_val, nodes_val, type_ids_val,
            global_type2id, node_map_val, num_rels, num_cats, use_edge_types
        )
        data_val = data_val.to(device)

        known_nodes_set = set(G_train.nodes())
        train_ids = [global_node2id.get(u, 0) for u in nodes_train]
        data_train.n_id = torch.tensor(train_ids, dtype=torch.long, device=device)
        val_ids = [global_node2id.get(u, 0) if u in known_nodes_set else 0 for u in nodes_val]
        data_val.n_id = torch.tensor(val_ids, dtype=torch.long, device=device)

        loaded_successfully = False
        if not TRAIN_MODE:
            if weight_path.exists():
                try:
                    Logger.info(f"[{val_p}] Loading weights from {weight_path}...")
                    # Ensure weights are loaded to the correct device map
                    model.load_state_dict(torch.load(weight_path, map_location=device))
                    loaded_successfully = True
                except Exception as e:
                    Logger.info(f"[{val_p}] Failed to load weights: {e}. Falling back to training.")
            else:
                Logger.info(f"[{val_p}] Weights not found. Falling back to training.")

        if not loaded_successfully:
            best_val_loss = float('inf')
            best_state = None
            patience_counter = 0

            for ep in range(epochs):
                train_loss, _ = train_one_epoch_generic(model, data_train, opt, G_train, idx_map_train, loss_type)
                val_loss = validate_one_epoch(model, data_val, G_val, idx_map_val, loss_type)

                if (ep+1) % 10 == 0:
                    Logger.info(f"[{FREQ}] period={val_p}, epoch={ep} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

                if val_loss < (best_val_loss - MIN_DELTA):
                    best_val_loss = val_loss.item()
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= PATIENCE:
                    Logger.info(f"[Early Stopping] No Val improvement for {PATIENCE} epochs.")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
                Logger.info(f"[{val_p}] Saving best model to {weight_path}")
                torch.save(best_state, weight_path)

        model.eval()
        with torch.no_grad(): emb = model(data_val)
        allowed = set(idx_map_val.keys())
        test_pos = [(u,v) for u,v in G_val.edges() if u in allowed and v in allowed]

        if test_pos:
            test_neg = sample_neg_edges(G_val, test_pos, k=NEG_PER_POS, allowed_nodes=allowed)
            metrics = evaluate_lp(emb, test_pos, test_neg, idx_map_val, known_nodes_set)

            for k, v in metrics.items():
                if k not in metrics_acc: metrics_acc[k] = []
                metrics_acc[k].append(v)

            msg = (f"[{FREQ}] {val_p} | ALL   (n={metrics.get('Count',0)}): "
                   f"MRR: {metrics.get('MRR',0):.4f}  | H@1: {metrics.get('Hits@1',0):.4f}  | H@3: {metrics.get('Hits@3',0):.4f}  | H@10: {metrics.get('Hits@10',0):.4f}")
            Logger.info(msg)

            if metrics.get("Seen_Count", 0) > 0:
                Logger.info(f"               | SEEN  (n={metrics['Seen_Count']}): MRR: {metrics.get('Seen_MRR',0):.4f}")
            if metrics.get("Unseen_Count", 0) > 0:
                Logger.info(f"               | NEW   (n={metrics['Unseen_Count']}): MRR: {metrics.get('Unseen_MRR',0):.4f}")
            Logger.info("-" * 60)
        else:
            for k in metrics_acc: metrics_acc[k].append(0.0)


        del data_train, data_val, G_train, G_val, X_train, X_val
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    avg_metrics = {k: np.mean(v) for k, v in metrics_acc.items() if v}
    return avg_metrics

def train_one_epoch_generic(model, data, optimizer, G_train, idx_map, loss_type="bpr"):
    model.train()
    emb = model(data)
    total_loss = torch.tensor(0.0, device=emb.device)

    pos_edges = list(G_train.edges())
    if pos_edges:
        neg_edges = sample_neg_edges(G_train, pos_edges, k=NEG_PER_POS, allowed_nodes=set(idx_map.keys()))
        if len(neg_edges) > 0:
            p_s, p_d = edge_index_from_pairs(pos_edges, idx_map)
            n_s, n_d = edge_index_from_pairs(neg_edges, idx_map)

            if loss_type == "findkg":
                total_loss += findkg_loss(emb, p_s, p_d, n_s, n_d)
            elif loss_type == "infonce" or loss_type == "hybrid":
                total_loss += info_nce_loss(emb, p_s, p_d, n_s, n_d, temperature=0.1)
            else:
                total_loss += bpr_link_loss(emb, p_s, p_d, n_s, n_d, margin=5.0)

    if loss_type in ["ce", "hybrid"]:
        logits = model.classify(emb)
        loss_ce = ce_node_loss(logits, data.y)
        total_loss += (HYBRID_CE_WEIGHT * loss_ce) if loss_type == "hybrid" else loss_ce

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total_loss.detach(), emb


def build_graph_from_df(df_slice, rel2id, rel_cat2id):
    G = nx.DiGraph()
    for _, r in df_slice.iterrows():
        u, v = r['sub'], r['obj']
        G.add_node(u, node_type=r.get('sub_type', 'UNK'))
        G.add_node(v, node_type=r.get('obj_type', 'UNK'))

        rel_id = rel2id.get(r['rel'], 0)
        cat_id = rel_cat2id.get(r['rel_category'], 0)

        G.add_edge(u, v,
                   rel_id=int(rel_id),
                   cat_id=int(cat_id),
                   w=float(r.get('w', 1.0)))
    return G

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT Link Prediction Experiments")
    parser.add_argument(
        "--strategy", 
        type=str, 
        choices=["original", "fallback", "strict", "all"],
        default="all",
        help="Which strategy to run: original, fallback, strict, or all (default: all)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="output/triplets_qwen2.5:14b_fnspid_JudgeLLM_metrics_computation.csv",
        help="Path to input data file"
    )
    args = parser.parse_args()
    
    data_file = Path(args.data)
    
    # Determine which strategies to run
    if args.strategy == "all":
        strategies = ["original", "fallback", "strict"]
    else:
        strategies = [args.strategy]

    if not data_file.exists():
        Logger.info("File not found.")
        exit(1)

    # REPORT DATASET STATISTICS
    Logger.info("\n" + "="*50)
    Logger.info("DATASET STATISTICS REPORT")
    Logger.info("="*50)
    stat_header = f"{'Strategy':<12} | {'Nodes':<8} | {'Edges':<8}"
    Logger.info(stat_header)
    Logger.info("-" * len(stat_header))

    for strategy in strategies:
        temp_df = load_df_from_csv(data_file, strategy=strategy)
        if temp_df.empty:
            Logger.info(f"{strategy:<12} | {'0':<8} | {'0':<8}")
            continue

        unique_nodes = set(temp_df['sub'].unique()) | set(temp_df['obj'].unique())
        num_edges = len(temp_df)
        Logger.info(f"{strategy:<12} | {len(unique_nodes):<8} | {num_edges:<8}")

    Logger.info("="*50 + "\n")

    for strategy in strategies:
        Logger.info("\n" + "#"*80)
        Logger.info(f"STARTING EXPERIMENT BATCH: STRATEGY = {strategy.upper()}")
        Logger.info("#"*80)

        df = load_df_from_csv(data_file, strategy=strategy)

        if DEBUG_N > 0:
            df = df.sort_values("date").iloc[:DEBUG_N]

        if df.empty:
            Logger.info(f"No valid data loaded for strategy {strategy}. Skipping.")
            continue

        global_type2id = build_type_map(df)
        all_nodes = set(df['sub'].unique()) | set(df['obj'].unique())
        global_node2id = {n: i+1 for i, n in enumerate(sorted(list(all_nodes)))}
        NUM_TOTAL_NODES = len(global_node2id) + 1
        Logger.info(f"Total Unique Nodes in History: {NUM_TOTAL_NODES}")

        base_configs = [
            {"base_name": "FindKG",  "loss": "findkg",  "epochs": 200, "lr": 1e-3},
            {"base_name": "BPR",     "loss": "bpr",     "epochs": 200, "lr": 1e-3},
            {"base_name": "CE",      "loss": "ce",      "epochs": 200, "lr": 1e-3},
            {"base_name": "Hybrid",  "loss": "hybrid",  "epochs": 100, "lr": 5e-4},
            {"base_name": "InfoNCE", "loss": "infonce", "epochs": 200, "lr": 1e-4},
        ]

        cases = []
        for conf in base_configs:
            # 1. Edge Features Included (Rel+Cat+NormW)
            cases.append({
                "name": f"{conf['base_name']} (Rel+Cat+NormW)",
                "fs_name": f"{conf['loss']}_edge_{strategy}",
                "edge_types": True,
                "loss": conf['loss'],
                "epochs": conf['epochs'],
                "learning_rate": conf['lr']
            })

            # 2. No Edge Features (Node Only)
            cases.append({
                "name": f"{conf['base_name']} (No Edge Feats)",
                "fs_name": f"{conf['loss']}_noedge_{strategy}",
                "edge_types": False,
                "loss": conf['loss'],
                "epochs": conf['epochs'],
                "learning_rate": conf['lr']
            })

        results = []
        for c in cases:
            metrics = run_experiment_case(df, c, global_type2id, global_node2id)
            results.append((c["name"], metrics))

        Logger.info("\n" + "="*80)
        Logger.info(f"{f'FINAL LEADERBOARD ({strategy.upper()})':^80}")
        Logger.info("="*80)
        results.sort(key=lambda x: x[1].get('MRR', 0), reverse=True)

        header = f"{'Rank':<4} | {'Model Name':<30} | {'MRR':<6} | {'H@1':<6} | {'H@3':<6} | {'H@10':<6}"
        Logger.info(header)
        Logger.info("-" * len(header))

        for rank, (name, m) in enumerate(results, 1):
            mrr = m.get('MRR', 0.0)
            h1 = m.get('Hits@1', 0.0)
            h3 = m.get('Hits@3', 0.0)
            h10 = m.get('Hits@10', 0.0)
            Logger.info(f"{rank:<4} | {name:<30} | {mrr:.4f} | {h1:.4f} | {h3:.4f} | {h10:.4f}")

        # Save results to CSV file
        results_data = []
        for rank, (name, m) in enumerate(results, 1):
            results_data.append({
                'Rank': rank,
                'Model': name,
                'Strategy': strategy,
                'MRR': m.get('MRR', 0.0),
                'Hits@1': m.get('Hits@1', 0.0),
                'Hits@3': m.get('Hits@3', 0.0),
                'Hits@10': m.get('Hits@10', 0.0),
                'Seen_MRR': m.get('Seen_MRR', 0.0),
                'Unseen_MRR': m.get('Unseen_MRR', 0.0)
            })
        
        results_df = pd.DataFrame(results_data)
        results_csv = Path(f"output/gat_results_{strategy}.csv")
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_csv, index=False)
        Logger.info(f"Results saved to: {results_csv}")

        Logger.info("\n" + "="*80 + "\n")
