import pandas as pd
import networkx as nx
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd



import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

def embed(G: nx.Graph, epochs: int = 400, embedding_dim: int = 16):
    """
    Learns node embeddings for graph G using a simple pairwise training scheme.
    
    This function:
      - Creates a learnable embedding for each node.
      - Constructs training pairs from edges (bidirectional) with their weights.
      - Uses a simple negative cosine similarity loss to push connected nodes closer.
      
    The graph is assumed to have unique keys for person nodes, with a 'type' attribute 
    (e.g., "person") and a 'name' attribute storing the person's actual name.
    
    Args:
        G (nx.Graph): The input graph.
        epochs (int): Number of training epochs.
        embedding_dim (int): Dimensionality of the learned embeddings.
        
    Returns:
        embeddings (nn.Embedding): The trained embedding layer.
    """
    # List all nodes and create a mapping to indices.
    node_list = list(G.nodes())
    num_nodes = len(node_list)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Create a learnable embedding layer.
    embeddings = nn.Embedding(num_nodes, embedding_dim)
    torch.manual_seed(42)  # For reproducibility
    
    # Build training pairs from graph edges (bidirectional).
    pairs = []
    for u, v, attr in G.edges(data=True):
        weight = attr.get("weight", 1.0)
        pairs.append((node_to_idx[u], node_to_idx[v], weight))
        pairs.append((node_to_idx[v], node_to_idx[u], weight))
    
    # Define a loss function using negative cosine similarity.
    def loss_fn(u_emb, v_emb, weight):
        cosine_sim = nn.functional.cosine_similarity(u_emb, v_emb, dim=0)
        return -weight * cosine_sim
    
    optimizer = optim.Adam(embeddings.parameters(), lr=0.01)
    
    # Training loop.
    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for u_idx, v_idx, weight in pairs:
            u_emb = embeddings(torch.tensor(u_idx))
            v_emb = embeddings(torch.tensor(v_idx))
            loss = loss_fn(u_emb, v_emb, weight)
            loss.backward()
            total_loss += loss.item()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss:.4f}")
    
    # Helper to get the unique node key for a person given their name.
    def get_person_key(person_name):
        for node, attr in G.nodes(data=True):
            if attr.get("type") == "person" and attr.get("name") == person_name:
                return node
        return None
    
    # Helper function to compute cosine similarity between two person nodes.
    def get_similarity(person1, person2):
        key1 = get_person_key(person1)
        key2 = get_person_key(person2)
        if key1 is None or key2 is None:
            return None
        idx1 = node_to_idx[key1]
        idx2 = node_to_idx[key2]
        emb1 = embeddings(torch.tensor(idx1))
        emb2 = embeddings(torch.tensor(idx2))
        return nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
    
    # Example similarity computations.
    sim_john_charlie = get_similarity("Joe", "Charlie")
    sim_john_janice = get_similarity("Joe", "Chuck")
    sim_john_bob = get_similarity("Joe", "Bob")
    sim_john_joe = get_similarity("Joe", "John")

    print("Similarity between John and Charlie:", sim_john_charlie)
    print("Similarity between John and Janice:", sim_john_janice)
    print("Similarity between John and Bob:", sim_john_bob)
    print("Similarity between John and Joe:", sim_john_joe)
    
    return embeddings

# Example usage:
if __name__ == '__main__':
    # Create a sample graph similar to what build_graph_from_df would produce.
    G = nx.Graph()
    # Person nodes: note the unique keys and attribute "name".
    G.add_node("person_John_0", type="person", name="John")
    G.add_node("person_Janice_1", type="person", name="Janice")
    G.add_node("person_Charlie_4", type="person", name="Charlie")
    # Trait nodes:
    G.add_node("MIT_undergrad", type="undergrad")
    G.add_node("Yale_grad", type="grad")
    G.add_node("Freelancer_employer", type="employer")
    
    # Edges for John.
    G.add_edge("person_John_0", "MIT_undergrad", weight=2.0)
    G.add_edge("person_John_0", "Yale_grad", weight=2.0)
    # Edge for Janice.
    G.add_edge("person_Janice_1", "Freelancer_employer", weight=2.0)
    # Connect Charlie similarly to John.
    G.add_edge("person_Charlie_4", "MIT_undergrad", weight=2.0)
    # Now learn embeddings.
    embed(G)
