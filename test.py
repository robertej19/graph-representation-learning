import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd





# --- 2. Learn Node Embeddings with PyTorch ---
# We create a learnable embedding for each node in our graph.
node_list = list(G.nodes())
num_nodes = len(node_list)
embedding_dim = 16

# Mapping from node to an index for our embedding matrix
node_to_idx = {node: idx for idx, node in enumerate(node_list)}

# Define a learnable embedding layer
embeddings = nn.Embedding(num_nodes, embedding_dim)
torch.manual_seed(42)  # For reproducibility

# Create training pairs from the graph edges (bidirectional)
pairs = []
for u, v, attr in G.edges(data=True):
    weight = attr.get("weight", 1.0)
    pairs.append((node_to_idx[u], node_to_idx[v], weight))
    pairs.append((node_to_idx[v], node_to_idx[u], weight))

# Define a simple loss that encourages connected nodes to have similar embeddings.
# We use negative cosine similarity (i.e. maximizing cosine similarity).
def loss_fn(u_emb, v_emb, weight):
    cosine_sim = nn.functional.cosine_similarity(u_emb, v_emb, dim=0)
    return -weight * cosine_sim

optimizer = optim.Adam(embeddings.parameters(), lr=0.01)
epochs = 100

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

# A helper function to compute similarity between two person nodes.
def get_similarity(person1, person2):
    idx1 = node_to_idx[person1]
    idx2 = node_to_idx[person2]
    emb1 = embeddings(torch.tensor(idx1))
    emb2 = embeddings(torch.tensor(idx2))
    return nn.functional.cosine_similarity(emb1, emb2, dim=0).item()

print("Similarity between John and Charlie:", get_similarity("John", "Charlie"))
print("Similarity between John and Janice:", get_similarity("John", "Janice"))

# --- 3. Visualize the Graph with Plotly and Dash ---
import plotly.graph_objects as go

# Use a spring layout for positioning nodes
pos = nx.spring_layout(G, seed=42)

# Build edge traces for Plotly
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Build node traces for Plotly
node_x = []
node_y = []
node_text = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_text,
    textposition="bottom center",
    marker=dict(
        size=10,
        color=[],
        colorscale='YlGnBu',
        colorbar=dict(
            thickness=15,
            title='Node Degree',
            xanchor='left',
            titleside='right'
        ),
        line_width=2)
)

# Color each node by its degree (number of connections)
node_adjacencies = []
for node in G.nodes():
    node_adjacencies.append(len(list(G.adj[node])))
node_trace.marker.color = node_adjacencies

# Create the Plotly figure
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Knowledge Graph Visualization',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40)
                ))
fig.show()

# --- 4. Display the Plotly Graph using Dash ---
import dash
from dash import dcc, html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Knowledge Graph Hybrid Visualization"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
