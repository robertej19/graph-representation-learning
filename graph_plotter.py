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

import networkx as nx
import plotly.graph_objects as go

def plot_graph(G: nx.Graph):
    """
    Plots the full undirected knowledge graph using Plotly.
    Uses a spring layout and then adjusts positions of person nodes to spread them out.
    Person nodes are rendered with larger markers and larger text font,
    while field nodes are rendered with smaller markers and text.
    """
    # Compute initial layout.
    pos = nx.spring_layout(G, seed=42)
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
        line=dict(width=1, color='#AAAAAA'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Separate nodes into "person" and "field" nodes.
    person_x, person_y, person_text, person_color = [], [], [], []
    field_x, field_y, field_text, field_color = [], [], [], []
    
    for node in G.nodes():
        x, y = pos[node]
        degree = len(list(G.adj[node]))
        node_type = G.nodes[node].get("type", "")
        if node_type == "person":
            person_x.append(x)
            person_y.append(y)
            # Use the 'name' attribute for person nodes.
            person_text.append(str(G.nodes[node].get("name", node)))
            person_color.append(degree)
        else:
            field_x.append(x)
            field_y.append(y)
            # Display node name along with type for field nodes.
            #field_text.append(f"{node} ({node_type})")
            field_text.append(f"{node} ")#({node_type})")
            field_color.append(degree)
    
    # Trace for person nodes: larger markers and larger text.
    person_trace = go.Scatter(
        x=person_x, y=person_y,
        mode='markers+text',
        text=person_text,
        textposition="bottom center",
        textfont=dict(size=20),  # larger text for people
        marker=dict(
            size=25,  # larger markers for people
            color=person_color,
            colorscale='YlGnBu',
            colorbar=dict(
                title='Node Degree',
                titleside='right'
            ),
            line_width=2
        ),
        hoverinfo='text'
    )
    
    # Trace for field nodes: smaller markers and text.
    field_trace = go.Scatter(
        x=field_x, y=field_y,
        mode='markers+text',
        text=field_text,
        textposition="bottom center",
        textfont=dict(size=10),  # smaller text for fields
        marker=dict(
            size=10,  # smaller markers for field nodes
            color=field_color,
            colorscale='YlGnBu',
            line_width=2
        ),
        hoverinfo='text'
    )
    
    # Create Plotly figure with dark gray theme.
    fig = go.Figure(data=[edge_trace, person_trace, field_trace],
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
