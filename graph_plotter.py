import plotly.graph_objects as go
import networkx as nx

import plotly.graph_objects as go
import networkx as nx

import plotly.graph_objects as go
import networkx as nx

def plot_graph(G: nx.Graph):
    # Use a spring layout for positioning nodes.
    pos = nx.spring_layout(G, seed=42)
    
    # Prepare edge data for Plotly.
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#AAAAAA'),
        hoverinfo='none',
        mode='lines'
    )

    # Prepare node data for Plotly.
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node} ({G.nodes[node].get('type', '')})")
    
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

    # Create the Plotly figure with a dark gray theme.
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Graph Visualization of People and Their Traits',
                        titlefont=dict(size=16, color='white'),
                        showlegend=False,
                        hovermode='closest',
                        paper_bgcolor='#2e2e2e',  # dark gray background for the paper
                        plot_bgcolor='#2e2e2e',   # dark gray background for the plot area
                        font=dict(color='white'),
                        margin=dict(b=20, l=5, r=5, t=40)
                    ))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    
    fig.show()


# Example usage:
if __name__ == '__main__':
    # Assuming G is your graph created from build_graph_from_df.
    print("this is where I would put a unit test \n \n IF I HAD ONE")
