import os
from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Optional, List, Set, cast
from pprint import pp
from job import Job
from skyfield.api import wgs84, load, EarthSatellite
from intervaltree import IntervalTree, Interval
from itree_render import render_tree_str
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


project_dir = Path(os.path.dirname(__file__))

data_dir = project_dir / 'data'

order_data_dir = data_dir / 'orders'

satellite_data_dir = data_dir / 'satellites'

jobs: List[Job] = []


def counter_generator():
    i = 0
    while True:
        yield i
        i += 1


counter = counter_generator()

for filename in os.listdir(order_data_dir):
    if filename.endswith('.json'):
        full_path = order_data_dir / filename
        with open(full_path, 'r') as f:
            data = json.load(f)
            job = Job(
                f'Job {next(counter)}',
                data['ImageStartTime'],
                data['ImageEndTime'],
                data['Priority'],
                data['Latitude'],
                data['Longitude']
            )
            jobs.append(job)

eph = load('de421.bsp')

ts = load.timescale()

t0 = ts.from_datetime(min(job.start for job in jobs).replace(tzinfo=timezone.utc))
t1 = ts.from_datetime(max(job.end for job in jobs).replace(tzinfo=timezone.utc))

satellites: List[EarthSatellite] = []

for filename in sorted(os.listdir(satellite_data_dir)):
    if filename.endswith('.json'):
        full_path = satellite_data_dir / filename
        with open(full_path, 'r') as f:
            data = json.load(f)
            sat = EarthSatellite(
                data['line1'],
                data['line2'],
                data['name'],
                ts
            )
            satellites.append(sat)

satellite = EarthSatellite(
    '1 00001U          23274.66666667  .00000000  00000-0  00000-0 0 00001',
    '2 00001 097.3597 167.6789 0009456 299.5645 340.3650 15.25701051000010',
    'SOSO-1',
    ts
)

altitude_degrees = 30.0
event_names = f'rise above {altitude_degrees}°', 'culminate', f'set below {altitude_degrees}°'

# This is an interval tree.
#
# Each node of the tree stores an interval, which has:
#   - start time
#   - end time
#   - data (containing the job that is able to be scheduled in this interval)
tree = IntervalTree()

trees = {sat: IntervalTree() for sat in satellites}

for sat in satellites:
    for job in jobs:
        # Location to be imaged
        location = wgs84.latlon(job.latitude, job.longitude)

        # Find all times that the location can be imaged by the satellite
        t, events = sat.find_events(
            location,
            t0,
            t1,
            altitude_degrees=altitude_degrees
        )

        # Check which of the imaging times are in sunlight
        sunlit = sat.at(t).is_sunlit(eph)

        for ti, event, sunlit_flag in zip(t, events, sunlit):
            if sunlit_flag:
                if event == 0:
                    start = ti.utc_datetime()
                elif event == 2:
                    if not start:
                        continue
                    end = ti.utc_datetime()

                    # satellite_job_slot = SatelliteJobSlot(sat, job)

                    # trees[sat].addi(start, end, set([satellite_job_slot]))
                    trees[sat].addi(start, end, set([job]))
                    start = None
                    end = None
            else:
                start = None
                end = None

# Merge overlaps in the interval tree.
#
# Whenever two intervals overlap, we create a new interval for them that
# accommodates them both, and the jobs that can be scheduled in that interval is
# the union of the jobs that could have been scheduled in the original
# intervals.
tree.merge_overlaps(data_reducer=lambda x, y: x.union(y))
for sat, tree in trees.items():
    tree.merge_overlaps(data_reducer=lambda x, y: x.union(y))

# pp(set([job_slot for interval in tree for job_slot in interval.data if job_slot.satellite.name == 'SOSO-1']))
# raise None

# print(render_tree_str(tree))
# print(len(tree))

# Create a directed graph
G = nx.DiGraph()

# Edges from source to each job
source_edges = [('source', job.name, 1) for job in jobs]

# Edges from jobs to timeslots.
# Remember: for each interval in the interval tree, we held the start and end
# time of the interval, along with all the jobs that could be scheduled in that
# interval.
#
# This for loop just creates edges between each job and the intervals that it
# could be scheduled in.
job_to_timeslot_edges = []
sat_edges = {sat: [] for sat in satellites}
for sat, tree in trees.items():
    for interval in tree:
        # print(f'INTERVAL: {interval}')
        for job in interval.data:
            # print(f'JOB: {job}')
            sat_edges[sat].append((job.name, f'{sat.name} {interval}', 1))
            job_to_timeslot_edges.append((job.name, f'{sat.name} {interval}', 1))

# Edges from source to each job
# sink_edges = [(interval, 'sink', 1) for interval in tree]
sink_edges = []
sat_to_sink_edges = {sat: [] for sat in satellites}

for sat, tree in trees.items():
    for interval in tree:
        sat_to_sink_edges[sat].append((f'{sat.name} {interval}', 'sink', 1))

sink_edges = [sat_to_sink_edge for sat_edges in sat_to_sink_edges.values() for sat_to_sink_edge in sat_edges]
edges = source_edges + job_to_timeslot_edges + sink_edges

# Add edges with capacities to the graph
for u, v, capacity in edges:
    G.add_edge(u, v, capacity=capacity)

# for node in G.nodes():
    # print(node)
# pp(list(G.nodes()))

# Define source and sink nodes
source = 'source'
sink = 'sink'

# Calculate the maximum flow and the flow on each edge
flow_value, flow_dict = nx.maximum_flow(G, source, sink)

# print("Max Flow Value:", flow_value)
# print("Flow on Each Edge:")
# for u in flow_dict:
    # for v, flow in flow_dict[u].items():
        # print(f"Edge {u} -> {v} has flow {flow}")

# Calculate the min cut using the minimum_cut function
cut_value, (reachable, non_reachable) = nx.minimum_cut(G, source, sink)
# print("Min Cut Value:", cut_value)
# print("Reachable Nodes in Min Cut:", reachable)
# print("Non-Reachable Nodes in Min Cut:", non_reachable)

fig, ax = plt.subplots()

pos = nx.multipartite_layout(
    G,
    subset_key={
        'a': ['source'],
        'b': [job.name for job in jobs],
        'c': [f'{sat.name} {interval}' for sat, tree in trees.items() for interval in tree],
        'd': ['sink']
    }
)
def array_op(x):
    # print(x)
    coords = pos[x]
    if x.startswith('Job'):
        return np.array((coords[0], coords[1]*5))
    else:
        return np.array((coords[0], coords[1]))
# array_op = lambda x: np.array((x[0]*2, x[1]))
pos = {p:array_op(p) for p in pos}
# for p in pos:
    # print(pos[p])
nx.draw_networkx_nodes(G, pos, nodelist=[job.name for job in jobs], node_color='red', node_size = 5)

nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-1'], node_color='blue', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-2'], node_color='purple', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-3'], node_color='pink', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-4'], node_color='yellow', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-5'], node_color='orange', node_size = 5)

nx.draw_networkx_nodes(G, pos, nodelist=['source'], node_color='green', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=['sink'], node_color='green', node_size = 5)
nx.draw_networkx_labels(G, pos, labels={'source': 'source', 'sink': 'sink'})
nx.draw_networkx_edges(G, pos, edgelist=source_edges, edge_color='red', arrows=False)
# nx.draw_networkx_edges(G, pos, edgelist=job_to_timeslot_edges, edge_color='blue', arrows=False)

nx.draw_networkx_edges(G, pos, edgelist=sat_edges[satellites[0]], edge_color='blue', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=sat_edges[satellites[1]], edge_color='purple', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=sat_edges[satellites[2]], edge_color='pink', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=sat_edges[satellites[3]], edge_color='yellow', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=sat_edges[satellites[4]], edge_color='orange', arrows=False)

nx.draw_networkx_edges(G, pos, edgelist=sat_to_sink_edges[satellites[0]], edge_color='blue', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=sat_to_sink_edges[satellites[1]], edge_color='purple', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=sat_to_sink_edges[satellites[2]], edge_color='pink', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=sat_to_sink_edges[satellites[3]], edge_color='yellow', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=sat_to_sink_edges[satellites[4]], edge_color='orange', arrows=False)

# nx.draw_networkx_edges(G, pos, edgelist=sink_edges, edge_color='green', arrows=False)

plt.title('All Possible Scheduling Opportunities')
# Create a custom legend
legend_elements = [
    plt.Line2D([0], [0], color='red', lw=2, label='Source to Jobs Connection'),
    plt.Line2D([0], [0], color='blue', lw=2, label='Job to Satellite Connection'),
    plt.Line2D([0], [0], color='blue', lw=2, label='Satellite to Sink Connection'),
    plt.Line2D([0], [0], marker='o', color='red', lw=0, markersize=5, label='Satellite Jobs'),
    plt.Line2D([0], [0], marker='o', color='blue', lw=0, markersize=5, label='SOSO-1 Slots'),
    plt.Line2D([0], [0], marker='o', color='purple', lw=0, markersize=5, label='SOSO-2 Slots'),
    plt.Line2D([0], [0], marker='o', color='pink', lw=0, markersize=5, label='SOSO-3 Slots'),
    plt.Line2D([0], [0], marker='o', color='yellow', lw=0, markersize=5, label='SOSO-4 Slots'),
    plt.Line2D([0], [0], marker='o', color='orange', lw=0, markersize=5, label='SOSO-5 Slots'),
]
# Add the legend to the axis
ax.legend(handles=legend_elements, loc='upper right')
plt.show()

fig, ax = plt.subplots()

new_source_edges = []
new_job_to_timeslot_edges = []
new_sink_edges = []
new_sat_to_sink_edges = {sat: [] for sat in satellites}
new_sat_edges = {sat: [] for sat in satellites}

for u in flow_dict:
    for v, flow in flow_dict[u].items():
        if isinstance(u, str) and u == 'source':
            if flow > 0:
                new_source_edges.append((u,v,1))
        elif isinstance(u, str) and u in [job.name for job in jobs]:
            if flow > 0:
                if v.startswith('SOSO-1'):
                    new_sat_edges[satellites[0]].append((u,v,1))
                elif v.startswith('SOSO-2'):
                    new_sat_edges[satellites[1]].append((u,v,1))
                elif v.startswith('SOSO-3'):
                    new_sat_edges[satellites[2]].append((u,v,1))
                elif v.startswith('SOSO-4'):
                    new_sat_edges[satellites[3]].append((u,v,1))
                elif v.startswith('SOSO-5'):
                    new_sat_edges[satellites[4]].append((u,v,1))
                new_job_to_timeslot_edges.append((u,v,1))
        elif isinstance(u, str) and u.startswith('SOSO'):
            if flow > 0:
                if u.startswith('SOSO-1'):
                    new_sat_to_sink_edges[satellites[0]].append((u,v,1))
                elif u.startswith('SOSO-2'):
                    new_sat_to_sink_edges[satellites[1]].append((u,v,1))
                elif u.startswith('SOSO-3'):
                    new_sat_to_sink_edges[satellites[2]].append((u,v,1))
                elif u.startswith('SOSO-4'):
                    new_sat_to_sink_edges[satellites[3]].append((u,v,1))
                elif u.startswith('SOSO-5'):
                    new_sat_to_sink_edges[satellites[4]].append((u,v,1))
                # new_job_to_timeslot_edges.append((u,v,1))
        # print(f"Edge {u} -> {v} has flow {flow}")

pos = nx.multipartite_layout(
    G,
    subset_key={
        'a': ['source'],
        'b': [job.name for job in jobs],
        'c': [f'{sat.name} {interval}' for sat, tree in trees.items() for interval in tree],
        'd': ['sink']
    }
)
pos = {p:array_op(p) for p in pos}
nx.draw_networkx_nodes(G, pos, nodelist=[job.name for job in jobs], node_color='red', node_size = 5)

nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-1'], node_color='blue', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-2'], node_color='purple', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-3'], node_color='pink', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-4'], node_color='yellow', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-5'], node_color='orange', node_size = 5)

nx.draw_networkx_nodes(G, pos, nodelist=['source'], node_color='green', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=['sink'], node_color='green', node_size = 5)
nx.draw_networkx_labels(G, pos, labels={'source': 'source', 'sink': 'sink'})
nx.draw_networkx_edges(G, pos, edgelist=new_source_edges, edge_color='red', arrows=False)
# nx.draw_networkx_edges(G, pos, edgelist=new_job_to_timeslot_edges, edge_color='blue', arrows=False)

nx.draw_networkx_edges(G, pos, edgelist=new_sat_edges[satellites[0]], edge_color='blue', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=new_sat_edges[satellites[1]], edge_color='purple', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=new_sat_edges[satellites[2]], edge_color='pink', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=new_sat_edges[satellites[3]], edge_color='yellow', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=new_sat_edges[satellites[4]], edge_color='orange', arrows=False)

nx.draw_networkx_edges(G, pos, edgelist=new_sat_to_sink_edges[satellites[0]], edge_color='blue', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=new_sat_to_sink_edges[satellites[1]], edge_color='purple', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=new_sat_to_sink_edges[satellites[2]], edge_color='pink', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=new_sat_to_sink_edges[satellites[3]], edge_color='yellow', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=new_sat_to_sink_edges[satellites[4]], edge_color='orange', arrows=False)

# nx.draw_networkx_edges(G, pos, edgelist=new_sink_edges, edge_color='green', arrows=False)
# Create a custom legend
legend_elements = [
    plt.Line2D([0], [0], color='red', lw=2, label='Source to Jobs Connection'),
    plt.Line2D([0], [0], color='blue', lw=2, label='Job to Satellite Connection'),
    plt.Line2D([0], [0], color='blue', lw=2, label='Satellite to Sink Connection'),
    plt.Line2D([0], [0], marker='o', color='red', lw=0, markersize=5, label='Satellite Jobs'),
    plt.Line2D([0], [0], marker='o', color='blue', lw=0, markersize=5, label='SOSO-1 Slots'),
    plt.Line2D([0], [0], marker='o', color='purple', lw=0, markersize=5, label='SOSO-2 Slots'),
    plt.Line2D([0], [0], marker='o', color='pink', lw=0, markersize=5, label='SOSO-3 Slots'),
    plt.Line2D([0], [0], marker='o', color='yellow', lw=0, markersize=5, label='SOSO-4 Slots'),
    plt.Line2D([0], [0], marker='o', color='orange', lw=0, markersize=5, label='SOSO-5 Slots'),
]
# Add the legend to the axis
ax.legend(handles=legend_elements, loc='upper right')
plt.title('Optimal Schedule')
plt.show()
