import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add edges along with their capacities
# For example, let's say we have a simple graph with edges and capacities:
# (source, target, capacity)
edges = [
    ('s', 'a', 10),
    ('s', 'b', 5),
    ('a', 'b', 15),
    ('a', 't', 10),
    ('b', 't', 10)
]

# Add edges with capacities to the graph
for u, v, capacity in edges:
    G.add_edge(u, v, capacity=capacity)

# Define source and sink nodes
source = 's'
sink = 't'

# Calculate the maximum flow and the flow on each edge
flow_value, flow_dict = nx.maximum_flow(G, source, sink)

print("Max Flow Value:", flow_value)
print("Flow on Each Edge:")
for u in flow_dict:
    for v, flow in flow_dict[u].items():
        print(f"Edge {u} -> {v} has flow {flow}")

# Calculate the min cut using the minimum_cut function
cut_value, (reachable, non_reachable) = nx.minimum_cut(G, source, sink)
print("Min Cut Value:", cut_value)
print("Reachable Nodes in Min Cut:", reachable)
print("Non-Reachable Nodes in Min Cut:", non_reachable)

print([u for u, v, capacity in edges if u not in ('s', 't')])

pos = nx.multipartite_layout(
    G,
    subset_key={
        'layer1': ['s'],
        'layer2': set(u for u, v, capacity in edges if u not in ('s', 't')),
        'layer3': ['t']
    }
)
nx.draw_networkx_nodes(G, pos, nodelist=[u for u, v, capacity in edges if u not in ('s', 't')], cmap=plt.get_cmap('jet'), node_color='red', node_size = 20)
nx.draw_networkx_nodes(G, pos, nodelist=['s'], cmap=plt.get_cmap('jet'), node_color='green', node_size = 20)
nx.draw_networkx_nodes(G, pos, nodelist=['t'], cmap=plt.get_cmap('jet'), node_color='green', node_size = 20)
# nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color='red', node_size = 5)
nx.draw_networkx_labels(G, pos, labels={'s': 'source', 't': 'sink'})
nx.draw_networkx_edges(G, pos, arrows=True)
# nx.draw_networkx_edges(G, pos, edgelist=new_source_edges, edge_color='red', arrows=False)
# nx.draw_networkx_edges(G, pos, edgelist=new_job_to_timeslot_edges, edge_color='blue', arrows=False)
# nx.draw_networkx_edges(G, pos, edgelist=new_sink_edges, edge_color='green', arrows=False)

plt.show()
