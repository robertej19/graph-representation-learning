import pandas as pd

# Read the CSV files.
data_df = pd.read_csv('data.csv')
weights_df = pd.read_csv('weights.csv')

# Define a list of colors to use for columns.
column_colors = ["#ff9999", "#99ff99", "#9999ff", "#ffcc99", "#cc99ff", "#99ffcc", "#ff99cc"]

def style_dataframe(df: pd.DataFrame) -> str:
    """Styles the DataFrame by coloring the text in each column with a unique color and returns HTML."""
    styled = df.head(10).style.set_table_attributes('class="csv-table"')
    for i, col in enumerate(df.columns):
        color = column_colors[i % len(column_colors)]
        # Set the text color for the column.
        styled = styled.set_properties(subset=[col], **{'color': color})
    return styled.to_html()

data_html = style_dataframe(data_df)
weights_html = style_dataframe(weights_df)

# Build an HTML page that includes the CSV previews and embeds the Plotly figures (assumed to be saved as HTML files).
html_template = f'''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Knowledge Graph Representation Learning Example - Basic</title>
  <style>
    body {{
      background-color: #2e2e2e;
      color: white;
      font-family: Arial, sans-serif;
      margin: 20px;
      line-height: 1.6;
    }}
    .container {{
      max-width: 1000px;
      margin: auto;
    }}
    h1, h2, h3 {{
      color: #f0f0f0;
    }}
    p {{
      margin-bottom: 20px;
    }}
    table.csv-table {{
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 40px;
    }}
    table.csv-table th, table.csv-table td {{
      border: 1px solid #555;
      padding: 8px;
      text-align: left;
    }}
    table.csv-table th {{
      background-color: #444;
    }}
    iframe {{
      width: 100%;
      height: 600px;
      border: none;
      margin-bottom: 40px;
    }}
    .section {{
      margin-bottom: 40px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Knowledge Graphs</h1>
    <p>
      We demonstrate how to use CSV data to build a knowledge graph and then learn node embeddings that capture the similarity between individuals.
      The process involves two major stages: First, we construct an <strong>undirected knowledge graph</strong> from our CSV files, where nodes represent people and their attributes, and edges capture relationships weighted by their importance.
      Then we learn vector embeddings for these nodes and reposition the person nodes in 2D space so that similar individuals are physically closer together, representing their mathematically derived relationships.
    </p>
    
    
    <div class="section">
      <h2>CSV Data</h2>
      <p>
        Here is a sample data set that we will use for the graphs that follow. <code>data.csv</code> contains details such as name, education, employer, location, hobbies, etc.,
        while <code>weights.csv</code> defines the relative importance of each field.
      </p>
      <h3>data.csv</h3>
      {data_html}
      <h3>weights.csv</h3>
      {weights_html}
    </div>
    
    <div class="section">
      <h2>Building the Undirected Knowledge Graph</h2>
      <p>
        Using the information from <code>data.csv</code> and <code>weights.csv</code>, we create an undirected graph.
        In this graph, each person and each trait (e.g., school, employer, hobby) is represented as a node.
        Edges connect people to their corresponding traits, with the edge weights reflecting the importance of that relationship. The physical distance between nodes is arbitrary, at this point, and closeness can only be understood by examining node degrees. 
      </p>
      <iframe src="undirected_graph.html"></iframe>
    </div>
    
    <div class="section">
      <h2>Learning Node Embeddings & Repositioning</h2>
      <p>
        After constructing the graph, we apply a machine learning process to learn vector embeddings for every node.
        These embeddings capture the similarity between nodes – so that individuals with similar attributes have embeddings that are close together.
        We then reposition the person nodes using Multi-Dimensional Scaling (MDS), resulting in a 2D layout where similar people are closer.
        Hovering over any person node in the graph reveals a list of other individuals sorted by similarity scores, with 1 being identical. In this example, where we put a heavy weight on location, we see Alice Joe and Bob, who all live in the same state, are very close together. Aime, who only shares a trait with Chuck, is far away from the pack.
      </p>
      <iframe src="embedded_graph.html"></iframe>
    </div>
    
    <div class="section">
      <h2>Summary</h2>
      <p>
        In summary, this visualization demonstrates the complete pipeline:
        <strong>CSV Data &rarr; Knowledge Graph &rarr; Node Embeddings &rarr; Interactive Visualization</strong>.
        The undirected graph shows the raw connections between people and their traits,
        while the embeddings graph provides a refined view based on learned similarities.
        Hover over nodes to explore the detailed similarity metrics.
      </p>
    </div>
  </div>
</body>
</html>
'''

# Write the final HTML to a file.
with open("final_page.html", "w") as f:
    f.write(html_template)
