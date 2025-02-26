import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.graph_objects as go

import torch
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import plotly.graph_objects as go

def reposition_people_by_embeddings(G: nx.Graph, embeddings: torch.nn.Embedding, node_to_idx: dict):
    """
    Repositions only the person nodes in graph G based on their learned embeddings.
    This version uses MDS to compute 2D positions that preserve the pairwise distances derived
    from cosine similarities between node embeddings.
    
    Args:
        G (nx.Graph): The full graph.
        embeddings (torch.nn.Embedding): The learned embedding layer.
        node_to_idx (dict): Mapping from node name to index in the embedding matrix.
    
    Returns:
        pos_dict (dict): A dictionary mapping person node keys to 2D positions.
    """
    # Filter for person nodes.
    people_nodes = [node for node, attr in G.nodes(data=True) if attr.get("type") == "person"]
    
    # Extract embeddings for these nodes.
    people_embeddings = []
    for node in people_nodes:
        idx = node_to_idx[node]
        emb = embeddings(torch.tensor(idx))
        people_embeddings.append(emb.detach().numpy())
    people_embeddings = np.array(people_embeddings)
    
    # Compute pairwise cosine similarities and convert to a distance matrix.
    # Cosine similarity ranges from -1 to 1, but typically our embeddings will have non-negative similarity.
    sim_matrix = cosine_similarity(people_embeddings)
    # Convert similarity to distance. A common choice is: distance = 1 - similarity.
    distance_matrix = 1 - sim_matrix
    
    # Use MDS to compute 2D positions that best preserve the pairwise distances.
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(distance_matrix)
    
    # Create a mapping from node to its 2D position.
    pos_dict = {node: pos for node, pos in zip(people_nodes, positions)}
    return pos_dict

def reposition_people_by_embeddings_tsne(G: nx.Graph, embeddings: torch.nn.Embedding, node_to_idx: dict):
    """
    Repositions only the person nodes in graph G based on their learned embeddings.
    It uses t-SNE to reduce embeddings to 2D positions.
    
    Args:
        G (nx.Graph): The full graph.
        embeddings (torch.nn.Embedding): The learned embedding layer.
        node_to_idx (dict): Mapping from node name to index in the embedding matrix.
    
    Returns:
        pos_dict (dict): A dictionary mapping person node keys to 2D positions.
    """
    # Filter for person nodes.
    people_nodes = [node for node, attr in G.nodes(data=True) if attr.get("type") == "person"]
    
    # Extract embeddings for these nodes.
    people_embeddings = []
    for node in people_nodes:
        idx = node_to_idx[node]
        emb = embeddings(torch.tensor(idx))
        people_embeddings.append(emb.detach().numpy())
    people_embeddings = np.array(people_embeddings)
    
    # Set perplexity to be less than the number of samples.
    perplexity = min(30, len(people_nodes) - 1) if len(people_nodes) > 1 else 1
    
    # Reduce to 2D using t-SNE.
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    positions = tsne.fit_transform(people_embeddings)
    
    # Create a mapping from node to its 2D position.
    pos_dict = {node: pos for node, pos in zip(people_nodes, positions)}
    return pos_dict
def plot_people_positions(pos_dict: dict, G: nx.Graph):
    """
    Plots only the person nodes with their 2D positions.
    
    Args:
        pos_dict (dict): Mapping from person node keys to 2D coordinates.
        G (nx.Graph): The graph (used to extract person names).
    """
    node_x = []
    node_y = []
    labels = []
    for node, pos in pos_dict.items():
        node_x.append(pos[0])
        node_y.append(pos[1])
        # Display the human-readable name stored in the "name" attribute.
        labels.append(G.nodes[node].get("name", node))
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=labels,
        textposition="bottom center",
        marker=dict(
            size=15,
            color='lightblue',
            line=dict(width=1, color='darkblue')
        )
    )
    
    fig = go.Figure(data=[node_trace],
                    layout=go.Layout(
                        title='People Positioned by Learned Embeddings',
                        titlefont=dict(size=16, color='white'),
                        paper_bgcolor='#2e2e2e',
                        plot_bgcolor='#2e2e2e',
                        font=dict(color='white'),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        margin=dict(b=20, l=5, r=5, t=40)
                    ))
    fig.show()


# Example usage:
if __name__ == '__main__':
    # For illustration, create a simple graph with person nodes and some trait nodes.
    G = nx.Graph()
    # Person nodes with unique keys and a "name" attribute.
    G.add_node("person_John_0", type="person", name="John")
    G.add_node("person_Janice_1", type="person", name="Janice")
    G.add_node("person_Charlie_4", type="person", name="Charlie")
    # Trait nodes (we won't reposition these here)
    G.add_node("MIT_undergrad", type="undergrad")
    G.add_node("Yale_grad", type="grad")
    G.add_node("Freelancer_employer", type="employer")
    
    # Add some edges.
    G.add_edge("person_John_0", "MIT_undergrad", weight=2.0)
    G.add_edge("person_John_0", "Yale_grad", weight=2.0)
    G.add_edge("person_Janice_1", "Freelancer_employer", weight=2.0)
    G.add_edge("person_Charlie_4", "MIT_undergrad", weight=2.0)
    
    pass
