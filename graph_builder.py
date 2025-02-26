import pandas as pd
import networkx as nx

def build_knowledge_graph_from_df(df: pd.DataFrame, df_weights: pd.DataFrame, default_weight: float = 1.0, sep: str = ";") -> nx.Graph:
    """
    Constructs an undirected graph from a people DataFrame.
    
    Each row in `df` represents a person and contains a "name" column plus other attributes.
    A unique key is generated for each person to avoid conflicts if names repeat.
    
    Trait nodes are created for every attribute (except "name"). If an attribute cell contains
    multiple values separated by `sep`, each value is processed separately.
    
    The df_weights DataFrame should have columns "field" and "weight". For each attribute (i.e., column name),
    if a weight is defined in df_weights, that weight is used; otherwise, the default_weight is used.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing people data.
        df_weights (pd.DataFrame): DataFrame with columns "field" and "weight" defining importance for each attribute.
        default_weight (float): Weight to use for attributes not found in df_weights.
        sep (str): Separator to split multi-valued attributes.
    
    Returns:
        nx.Graph: An undirected graph with person and trait nodes, and weighted edges connecting them.
    """
    G = nx.Graph()
    
    # Create a dictionary mapping field names to their corresponding weight.
    weight_map = dict(zip(df_weights["field"], df_weights["weight"]))
    
    # Iterate over each person in the DataFrame.
    for idx, row in df.iterrows():
        # Create a unique key for the person using the name and row index.
        person_key = f"{row['name']}_{idx}"
        # Add the person node (store the original name as an attribute).
        G.add_node(person_key, type="person", name=row["name"])
        
        # Process every column except "name".
        for col in df.columns:
            if col == "name":
                continue
            
            value = row[col]
            # Skip if the value is missing.
            if pd.isnull(value):
                continue
            
            # If the cell is a string and contains the separator, split into multiple values.
            if isinstance(value, str) and sep in value:
                values = [v.strip() for v in value.split(sep) if v.strip()]
            else:
                # Otherwise, treat the cell as a single value.
                values = [str(value)]
            
            # For each value in the cell, add a trait node and an edge.
            for v in values:
                # Namespacing the trait by the column name (so "MIT" under "undergrad" is different from "MIT" elsewhere).
                trait_node = f"{v}"
                if not G.has_node(trait_node):
                    G.add_node(trait_node, type=col)
                # Get the weight from df_weights if defined, else use the default.
                weight = weight_map.get(col, default_weight)
                G.add_edge(person_key, trait_node, weight=weight)
    
    return G


