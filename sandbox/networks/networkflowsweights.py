import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add supply nodes (tasks) with supply values
tasks = ["T1", "T2", "T3"]
for task in tasks:
    # G.add_node(task, demand=-1)  # -1 means each task needs to be scheduled once
    G.add_node(task)

# Add demand nodes (rooms) with demand values
rooms = ["R1", "R2"]
room_capacities = {"R1": 2, "R2": 1}  # Room capacities
for room, capacity in room_capacities.items():
    G.add_node(room, demand=capacity)  # Positive demand indicates room capacity

# Add the source and sink nodes
G.add_node("source", demand=-len(tasks))  # Total supply is the number of tasks
G.add_node("sink", demand=len(tasks))     # Total demand is the same

# Connect source to tasks with zero-cost edges
for task in tasks:
    G.add_edge("source", task, capacity=1, weight=0)

# Connect rooms to sink with zero-cost edges
for room in rooms:
    G.add_edge(room, "sink", capacity=room_capacities[room], weight=0)

# Connect tasks to rooms with costs (based on priorities)
# Lower cost for higher priority
priority_costs = {
    ("T1", "R1"): 1,
    ("T1", "R2"): 3,
    ("T2", "R1"): 2,
    ("T2", "R2"): 1,
    ("T3", "R1"): 3,
    ("T3", "R2"): 2,
}
for (task, room), cost in priority_costs.items():
    G.add_edge(task, room, capacity=1, weight=cost)

# Compute the minimum cost flow
flow_cost, flow_dict = nx.network_simplex(G)

print("Total Minimum Cost:", flow_cost)
print("Flow Assignment:")
for node, flows in flow_dict.items():
    for target, flow in flows.items():
        if flow > 0 and node in tasks:
            print(f"Task {node} assigned to Room {target}")
