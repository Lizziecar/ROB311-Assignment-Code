import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Define vertices and edges
V = np.arange(0, 10)
E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])

# Create a graph
G = nx.Graph()

# Add nodes
for v in V:
    G.add_node(v)

# Add edges
for edge in E:
    G.add_edge(edge[0], edge[1])

# Draw the graph
nx.draw(G, with_labels=True)
plt.show()
