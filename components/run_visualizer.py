import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import torch
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

from components.gat_model import GATLP
from utils.data_utils import load_df_from_csv
from utils.visualization_utils import build_networkx_graph, save_pyvis_html, prepare_graph_data
from utils.gat_utils import build_node_features, pack_pyg_data

ALL_STRATEGIES = ["original", "fallback", "strict"]
ALL_LOSSES = ["findkg", "bpr", "ce", "infonce", "hybrid"]
ALL_EDGE_MODES = [True, False] # True = With Rel Types, False = No Edge Feats

def build_nx_from_df_slice(df_slice, rel2id, cat2id):
    """Builds a temporary NetworkX graph for feature engineering extraction."""
    G = nx.DiGraph()
    for _, r in df_slice.iterrows():
        u, v = r['sub'], r['obj']
        G.add_node(u, node_type=r.get('sub_type', 'UNK'))
        G.add_node(v, node_type=r.get('obj_type', 'UNK'))
        G.add_edge(u, v,
                   rel_id=rel2id.get(r['rel'], 0),
                   cat_id=cat2id.get(r['rel_category'], 0),
                   w=float(r.get('w', 1.0)))
    return G

def build_maps(df):
    """Recreates the global ID maps used during training."""
    rels = sorted(df['rel'].dropna().unique())
    rel2id = {r: i for i, r in enumerate(rels)}

    cats = set(df['rel_category'].dropna().unique())
    cats.add("UNK")
    cat2id = {c: i for i, c in enumerate(sorted(list(cats)))}

    types = set(df['sub_type'].dropna().unique()) | set(df['obj_type'].dropna().unique())
    types.add("UNK")
    type2id = {t: i for i, t in enumerate(sorted(list(types)))}

    all_nodes = set(df['sub'].unique()) | set(df['obj'].unique())
    node2id = {n: i+1 for i, n in enumerate(sorted(list(all_nodes)))}

    return rel2id, cat2id, type2id, node2id

def main():
    parser = argparse.ArgumentParser(description="Batch Generator for KG Visualizations (Optimized)")

    parser.add_argument("--triplets_path", type=str, default="output/triplets_qwen2.5:32b_fnspid_JudgeLLM_metrics_computation.csv",
                        help="Path to the .csv file containing triplets")
    parser.add_argument("--weights_dir", type=str, default="weights",
                        help="Directory where gat.py weights are saved")
    parser.add_argument("--output_dir", type=str, default="output/kg_visualization",
                        help="Root directory for outputs")

    args = parser.parse_args()
    # Use CPU for viz to avoid OOM on large batches, or cuda if available and graph is small
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    for strategy in ALL_STRATEGIES:
        print(f"\nLoading strategy: {strategy.upper()}")

        df_master = load_df_from_csv(args.triplets_path, strategy=strategy)
        if df_master.empty:
            print("Empty dataframe.")
            continue

        rel2id, cat2id, type2id, node2id = build_maps(df_master)
        num_global_nodes = len(node2id) + 1

        if 'date' not in df_master.columns: continue
        df_master['period'] = pd.to_datetime(df_master['date']).dt.to_period('M').astype(str)
        periods = sorted(df_master['period'].unique())

        print(f"Generating RAW (No-Prediction) Monthly Plots...")
        raw_out_dir = os.path.join(args.output_dir, strategy, "raw_no_prediction")
        os.makedirs(raw_out_dir, exist_ok=True)
        for period in periods:
             subset = df_master[df_master['period'] == period]
             nodes, edges = prepare_graph_data(subset, freq='M')
             if 'score' in edges.columns: del edges['score']
             G = build_networkx_graph(nodes, edges)
             save_pyvis_html(G, os.path.join(raw_out_dir, f"graph_{period}_raw.html"))

        print("Initializing models...")
        dummy_G = build_nx_from_df_slice(df_master.iloc[:5], rel2id, cat2id)
        dummy_node_map = {u: dummy_G.nodes[u].get("node_type", "UNK") for u in dummy_G.nodes()}
        dummy_X, _, _ = build_node_features(dummy_G, dummy_node_map, type2id)
        in_ch = dummy_X.shape[1]

        model_edge = GATLP(in_ch=in_ch, hid=256, num_classes=len(type2id),
                           num_global_nodes=num_global_nodes, num_rels=len(rel2id),
                           num_cats=len(cat2id), use_edge_types=True).to(device)

        model_noedge = GATLP(in_ch=in_ch, hid=256, num_classes=len(type2id),
                             num_global_nodes=num_global_nodes, num_rels=len(rel2id),
                             num_cats=len(cat2id), use_edge_types=False).to(device)

        print("Processing months & scoring...")
        for period in tqdm(periods, desc=f"  Strategy: {strategy}"):
            month_df = df_master[df_master['period'] == period].copy()
            if month_df.empty: continue

            G_month = build_nx_from_df_slice(month_df, rel2id, cat2id)
            n_map = {u: G_month.nodes[u].get("node_type", "UNK") for u in G_month.nodes()}
            X, nodes_list, type_ids = build_node_features(G_month, n_map, type2id)

            local_node_to_idx = {u: i for i, u in enumerate(nodes_list)}
            global_ids = torch.tensor([node2id.get(u, 0) for u in nodes_list], dtype=torch.long)

            # Create index tensors for all edges in this month
            src_indices = [local_node_to_idx.get(u) for u in month_df['sub']]
            dst_indices = [local_node_to_idx.get(v) for v in month_df['obj']]

            # Filter valid (should be all, but safe check)
            valid_mask = [i is not None and j is not None for i, j in zip(src_indices, dst_indices)]
            if not any(valid_mask): continue

            month_df = month_df[valid_mask] # Sync df
            src_tensor = torch.tensor([i for i, m in zip(src_indices, valid_mask) if m], dtype=torch.long, device=device)
            dst_tensor = torch.tensor([i for i, m in zip(dst_indices, valid_mask) if m], dtype=torch.long, device=device)

            # Pre-pack PyG data objects for both architectures
            data_edge, _ = pack_pyg_data(G_month, X, nodes_list, type_ids, type2id, n_map, len(rel2id), len(cat2id), use_edge_types=True)
            data_edge.n_id = global_ids

            data_noedge, _ = pack_pyg_data(G_month, X, nodes_list, type_ids, type2id, n_map, len(rel2id), len(cat2id), use_edge_types=False)
            data_noedge.n_id = global_ids

            # B. Iterate All Models for this Month
            for loss_type in ALL_LOSSES:
                for use_edge_types in ALL_EDGE_MODES:
                    # Configuration Setup
                    feat_str = "edge" if use_edge_types else "noedge"
                    weight_folder = f"{loss_type}_{feat_str}_{strategy}"
                    w_path = os.path.join(args.weights_dir, weight_folder, f"{period}.pt")

                    if not os.path.exists(w_path): continue

                    # Select pre-loaded model & data
                    model = model_edge if use_edge_types else model_noedge
                    data = data_edge if use_edge_types else data_noedge

                    try:
                        # Load Weights
                        model.load_state_dict(torch.load(w_path, map_location=device))
                        model.eval()

                        # Inference
                        with torch.no_grad():
                            emb = model(data.to(device)) # [NumNodes, Hidden]

                            # Vectorized Score Calculation
                            # Gather embeddings for src and dst
                            # (Batch Size, Hidden) * (Batch Size, Hidden) -> Sum -> (Batch Size)
                            s_src = emb[src_tensor]
                            s_dst = emb[dst_tensor]
                            scores = (s_src * s_dst).sum(dim=1).cpu().numpy()

                        pred_out_dir = os.path.join(args.output_dir, strategy, f"pred_{loss_type}_{feat_str}")
                        os.makedirs(pred_out_dir, exist_ok=True)

                        viz_df = month_df.copy()
                        viz_df['score'] = scores

                        nodes, edges = prepare_graph_data(viz_df, freq='M')
                        edges['score'] = viz_df['score'].values # Transfer scores

                        G_viz = build_networkx_graph(nodes, edges)
                        save_pyvis_html(G_viz, os.path.join(pred_out_dir, f"graph_{period}_scored.html"))

                    except Exception as e:
                        print(f"Error processing {period} {loss_type}: {e}")
                        pass

if __name__ == "__main__":
    main()
