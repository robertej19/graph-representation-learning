import plotly.graph_objects as go
import networkx as nx

import plotly.graph_objects as go
import networkx as nx

import plotly.graph_objects as go
import networkx as nx
import numpy as np

def adjust_person_positions(pos: dict, G: nx.Graph, min_dist: float = 0.3, iterations: int = 50, adjustment_factor: float = 0.1):
    """
    Adjusts positions for nodes of type "person" so that they are spread out.
    Only the positions for person nodes are modified; positions of trait nodes remain fixed.
    
    Args:
        pos (dict): Mapping from node to 2D position (e.g., from spring_layout).
        G (nx.Graph): The input graph.
        min_dist (float): Minimum allowed distance between person nodes.
        iterations (int): Number of adjustment iterations.
        adjustment_factor (float): Scaling factor for how much to nudge nodes.
        
    Returns:
        dict: Updated position mapping.
    """
    # Get list of person nodes.
    person_nodes = [node for node, attr in G.nodes(data=True) if attr.get("type") == "person"]
    # Convert positions of person nodes to an array.
    positions = np.array([pos[node] for node in person_nodes])
    
    # Perform iterative repulsion adjustment.
    for _ in range(iterations):
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    if dist == 0:
                        diff = np.random.rand(2) - 0.5
                        dist = np.linalg.norm(diff)
                    # Compute how much to adjust.
                    adjustment = adjustment_factor * (min_dist - dist)
                    delta = adjustment * (diff / dist)
                    positions[i] += delta
                    positions[j] -= delta
    # Update the original pos mapping for person nodes.
    for i, node in enumerate(person_nodes):
        pos[node] = positions[i]
    return pos

def plot_graph(G: nx.Graph):
    """
    Plots the full undirected knowledge graph using Plotly.
    Uses a spring layout and then adjusts positions of person nodes to spread them out.
    """
    # Compute initial layout.
    pos = nx.spring_layout(G, seed=42)
    # Adjust positions for "person" nodes to reduce overlap.
    pos = adjust_person_positions(pos, G, min_dist=0.3, iterations=50, adjustment_factor=0.1)
    
    # Build edge traces.
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#AAAAAA'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Build node traces.
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        #node_text.append(f"{node} ({G.nodes[node].get('type', '')})")
        node_text.append(f"{node}")# ({G.nodes[node].get('type', '')})")
    
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
                title='Node Degree',
                titleside='right'
            ),
            line_width=2
        )
    )
    
    # Color nodes by degree.
    node_adjacencies = [len(list(G.adj[node])) for node in G.nodes()]
    node_trace.marker.color = node_adjacencies
    
    # Create Plotly figure with a dark gray theme.
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Graph Visualization of People and Their Traits',
                        titlefont=dict(size=16, color='white'),
                        showlegend=False,
                        hovermode='closest',
                        paper_bgcolor='#2e2e2e',
                        plot_bgcolor='#2e2e2e',
                        font=dict(color='white'),
                        margin=dict(b=20, l=5, r=5, t=40)
                    ))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    return fig
# Example usage:
if __name__ == '__main__':
    # Assuming G is your graph created from build_graph_from_df.
    print("this is where I would put a unit test \n \n IF I HAD ONE")
