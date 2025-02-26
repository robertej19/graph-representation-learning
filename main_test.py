import pandas as pd
import networkx as nx
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd



from graph_builder import build_knowledge_graph_from_df
from graph_plotter import plot_graph
from machine_modeling import embed
from plotting_embedded import reposition_people_by_embeddings, adjust_positions
from plotting_embedded import plot_people_positions, compute_hover_texts
# Example usage:
if __name__ == "__main__":
    # --- 0. Load the data ---
    df = pd.read_csv("data.csv")
    df_weights = pd.read_csv("weights.csv")
    
    # --- 1. Build the Knowledge Graph ---
    G = build_knowledge_graph_from_df(df, df_weights)
    
    # --- 2. Display the Graph ---
    fig_undirected = plot_graph(G)
    fig_undirected.write_html("undirected_graph.html")

    e = embed(G)

    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Obtain initial positions based on embeddings.
    pos_dict, people_nodes, people_embs = reposition_people_by_embeddings(G, e, node_to_idx)
    
    # Adjust positions to avoid overlaps.
    #pos_dict = adjust_positions(pos_dict, min_dist=0.2, iterations=20, adjustment_factor=0.05)
    
    # Compute hover texts showing similarity scores.
    hover_texts = compute_hover_texts(people_nodes, people_embs, G, node_to_idx)
    
    # Plot the adjusted positions with hover information.
    fig_embedded = plot_people_positions(pos_dict, G, people_nodes, hover_texts)
    

    fig_embedded.write_html("embedded_graph.html")
        