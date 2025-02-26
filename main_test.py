import pandas as pd
import networkx as nx

from graph_builder import build_knowledge_graph_from_df
from graph_plotter import plot_graph

# Example usage:
if __name__ == "__main__":
    # --- 0. Load the data ---
    df = pd.read_csv("sample_data.csv")
    df_weights = pd.read_csv("weights.csv")
    
    # --- 1. Build the Knowledge Graph ---
    G = build_knowledge_graph_from_df(df, df_weights)
    
    # --- 2. Display the Graph ---
    plot_graph(G)