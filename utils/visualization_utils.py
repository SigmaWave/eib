import networkx as nx
import pandas as pd
import os
from pyvis.network import Network
from tqdm import tqdm

def build_networkx_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    for _, row in tqdm(nodes_df.iterrows(), total=nodes_df.shape[0], desc="Building Nodes"):
        node_id = str(row["id"])
        label = row.get("label", node_id)
        n_type = row.get("type", "UNK")
        time_val = str(row.get("time", "ALL"))

        G.add_node(
            node_id,
            label=label,
            title=f"Type: {n_type}, Time: {time_val}",
            group=time_val,
            node_type=n_type
        )

    for _, row in tqdm(edges_df.iterrows(), total=edges_df.shape[0], desc="Building Edges"):
        src = str(row["source"])
        dst = str(row["target"])
        rel = row.get("label", "RELATED_TO")
        time_val = str(row.get("time", "ALL"))

        G.add_edge(
            src,
            dst,
            label=rel,
            title=f"Rel: {rel}, Time: {time_val}",
            group=time_val
        )

    return G

def generate_monthly_plots(df: pd.DataFrame, output_dir: str):
    if 'date' not in df.columns:
        print("Date column missing, skipping monthly plots.")
        return

    df['period'] = pd.to_datetime(df['date']).dt.to_period('M').astype(str)
    periods = sorted(df['period'].unique())

    for period in tqdm(periods, desc="Generating Monthly HTMLs"):
        subset = df[df['period'] == period]
        if subset.empty: continue

        nodes, edges = prepare_graph_data(subset, freq='M')
        G = build_networkx_graph(nodes, edges)

        path = os.path.join(output_dir, f"monthly_graphs/graph_{period}.html")
        save_pyvis_html(G, path)

def prepare_graph_data(df: pd.DataFrame, time_col: str = 'date', freq: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = df[['sub', 'sub_type']].rename(columns={'sub': 'id', 'sub_type': 'type'})
    objects = df[['obj', 'obj_type']].rename(columns={'obj': 'id', 'obj_type': 'type'})

    nodes = pd.concat([subjects, objects]).drop_duplicates(subset=['id']).reset_index(drop=True)
    nodes['label'] = nodes['id']

    edges = df.rename(columns={'sub': 'source', 'obj': 'target', 'rel': 'label'})

    if freq and time_col in df.columns:
        time_labels = pd.to_datetime(df[time_col]).dt.to_period(freq).astype(str)
        edges['time'] = time_labels
        nodes['time'] = 'ALL'
    else:
        edges['time'] = 'ALL'
        nodes['time'] = 'ALL'

    return nodes, edges

def save_pyvis_html(G: nx.DiGraph, output_path: str, filter_physics: bool = True, notebook: bool = False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    net = Network(
        notebook=notebook,
        cdn_resources="remote",
        bgcolor="#222222",
        font_color="white",
        height="750px",
        width="100%",
        select_menu=True,
        filter_menu=True,
    )

    net.from_nx(G)

    if filter_physics:
        net.show_buttons(filter_=['physics'])

    try:
        net.save_graph(output_path)
    except Exception as e:
        print(f"Error saving graph to {output_path}: {e}")
