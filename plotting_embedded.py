import torch
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import plotly.graph_objects as go

def reposition_people_by_embeddings(G: nx.Graph, embeddings: torch.nn.Embedding, node_to_idx: dict):
    """
    Computes 2D positions for person nodes using MDS on the distance matrix (1 - cosine similarity).
    
    Args:
        G (nx.Graph): The full graph.
        embeddings (torch.nn.Embedding): The learned embedding layer.
        node_to_idx (dict): Mapping from node name to index.
        
    Returns:
        pos_dict (dict): Mapping of person node to 2D position.
        people_nodes (list): List of person node names.
        people_embs (np.array): Array of person embeddings.
    """
    # Filter person nodes.
    people_nodes = [node for node, attr in G.nodes(data=True) if attr.get("type") == "person"]
    
    # Extract embeddings.
    people_embs = []
    for node in people_nodes:
        idx = node_to_idx[node]
        emb = embeddings(torch.tensor(idx))
        people_embs.append(emb.detach().numpy())
    people_embs = np.array(people_embs)
    
    # Compute pairwise cosine similarity and convert to distance.
    sim_matrix = cosine_similarity(people_embs)
    distance_matrix = 1 - sim_matrix
    
    # Use MDS to get 2D positions.
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(distance_matrix)
    
    # Build initial position dict.
    pos_dict = {node: pos for node, pos in zip(people_nodes, positions)}
    return pos_dict, people_nodes, people_embs

def adjust_positions(pos_dict: dict, min_dist: float = 0.2, iterations: int = 20, adjustment_factor: float = 0.005):
    """
    Adjusts the positions so that nodes that are too close are pushed apart.
    
    Args:
        pos_dict (dict): Mapping from node to 2D position (np.array).
        min_dist (float): Minimum allowed distance between any two nodes.
        iterations (int): How many passes to adjust.
        adjustment_factor (float): Scaling factor for adjustments.
    
    Returns:
        new_pos_dict (dict): Adjusted positions.
    """
    # Convert dictionary to array.
    nodes = list(pos_dict.keys())
    positions = np.array([pos_dict[node] for node in nodes])
    
    for _ in range(iterations):
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    # Avoid division by zero.
                    if dist == 0:
                        diff = np.random.rand(2) - 0.5
                        dist = np.linalg.norm(diff)
                    # Compute adjustment magnitude.
                    adjustment = adjustment_factor * (min_dist - dist)
                    # Move nodes in opposite directions.
                    delta = adjustment * (diff / dist)
                    positions[i] += delta
                    positions[j] -= delta
    new_pos_dict = {node: pos for node, pos in zip(nodes, positions)}
    return new_pos_dict

def compute_hover_texts(people_nodes, people_embs, G, node_to_idx):
    """
    For each person, computes a hover text string that includes cosine similarity scores
    with all other person nodes, sorted by similarity (closest first).
    
    Args:
        people_nodes (list): List of person node keys.
        people_embs (np.array): Array of embeddings for these nodes.
        G (nx.Graph): The full graph.
        node_to_idx (dict): Mapping from node key to embedding index.
    
    Returns:
        hover_texts (dict): Mapping from node key to hover text string.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    # Compute the full cosine similarity matrix.
    sim_matrix = cosine_similarity(people_embs)
    hover_texts = {}
    
    for i, node in enumerate(people_nodes):
        person_name = G.nodes[node].get("name", node)
        # Build a list of (other_person_name, similarity) for every other node.
        sim_list = []
        for j, other_node in enumerate(people_nodes):
            if i == j:
                continue
            other_name = G.nodes[other_node].get("name", other_node)
            sim = sim_matrix[i, j]
            sim_list.append((other_name, sim))
        # Sort the list in descending order (highest similarity first)
        sim_list_sorted = sorted(sim_list, key=lambda x: x[1], reverse=True)
        # Build hover text: first line is the person's own name, then each other person and its similarity.
        text_lines = [f"<b>{person_name}</b>"]
        for other_name, sim in sim_list_sorted:
            text_lines.append(f"{other_name}: {sim:.2f}")
        hover_texts[node] = "<br>".join(text_lines)
    return hover_texts


def plot_people_positions(pos_dict: dict, G: nx.Graph, people_nodes, hover_texts: dict):
    """
    Plots the person nodes at positions given by pos_dict.
    Hovering over a node shows its similarity scores with all other people.
    
    Args:
        pos_dict (dict): Mapping from person node to 2D position.
        G (nx.Graph): The graph (for obtaining names).
        people_nodes (list): List of person node keys.
        hover_texts (dict): Mapping from person node to hover text.
    """
    node_x = []
    node_y = []
    texts = []
    for node in people_nodes:
        pos = pos_dict[node]
        node_x.append(pos[0])
        node_y.append(pos[1])
        texts.append(hover_texts[node])
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[G.nodes[node].get("name", node) for node in people_nodes],
        hovertext=texts,
        hoverinfo='text',
        textposition="bottom center",
        marker=dict(
            size=15,
            color='lightblue',
            line=dict(width=1, color='darkblue')
        )
    )
    
    fig = go.Figure(data=[node_trace],
                    layout=go.Layout(
                        title='People Positioned by Learned Embeddings (Adjusted)',
                        titlefont=dict(size=16, color='white'),
                        paper_bgcolor='#2e2e2e',
                        plot_bgcolor='#2e2e2e',
                        font=dict(color='white'),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        margin=dict(b=20, l=5, r=5, t=40)
                    ))
    return fig

# Example usage:
if __name__ == '__main__':
    # Create a sample graph.
    G = nx.Graph()
    # Person nodes with unique keys and "name" attribute.
    G.add_node("person_John_0", type="person", name="John")
    G.add_node("person_Janice_1", type="person", name="Janice")
    G.add_node("person_Charlie_4", type="person", name="Charlie")
    # Some trait nodes.
    G.add_node("MIT_undergrad", type="undergrad")
    G.add_node("Yale_grad", type="grad")
    G.add_node("Freelancer_employer", type="employer")
    
    # Edges.
    G.add_edge("person_John_0", "MIT_undergrad", weight=2.0)
    G.add_edge("person_John_0", "Yale_grad", weight=2.0)
    G.add_edge("person_Janice_1", "Freelancer_employer", weight=2.0)
    G.add_edge("person_Charlie_4", "MIT_undergrad", weight=2.0)
    
    # Build the node-to-index mapping.
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # For demonstration, create a dummy embedding layer.
    embedding_dim = 16
    num_nodes = len(node_list)
    torch.manual_seed(42)
    dummy_embeddings = torch.nn.Embedding(num_nodes, embedding_dim)
    
