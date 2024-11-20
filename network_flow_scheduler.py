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


project_dir = Path(os.path.dirname(__file__))

data_dir = project_dir / 'data'

jobs: List[Job] = []


def counter_generator():
    i = 0
    while True:
        yield i
        i += 1


counter = counter_generator()

for filename in os.listdir(data_dir):
    if filename.endswith('.json'):
        full_path = data_dir / filename
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

satellite = EarthSatellite(
    '1 00001U          23274.66666667  .00000000  00000-0  00000-0 0 00001',
    '2 00001 097.3597 167.6789 0009456 299.5645 340.3650 15.25701051000010',
    'SOSO-1',
    ts
)

altitude_degrees = 30.0
event_names = f'rise above {altitude_degrees}°', 'culminate', f'set below {altitude_degrees}°'

tree = IntervalTree()

for job in jobs:
    # Location to be imaged
    location = wgs84.latlon(job.latitude, job.longitude)

    # Find all times that the location can be imaged by the satellite
    t, events = satellite.find_events(
        location,
        t0,
        t1,
        altitude_degrees=altitude_degrees
    )

    # Check which of the imaging times are in sunlight
    sunlit = satellite.at(t).is_sunlit(eph)

    for ti, event, sunlit_flag in zip(t, events, sunlit):
        if sunlit_flag:
            if event == 0:
                start = ti.utc_datetime()
            elif event == 2:
                if not start:
                    continue
                end = ti.utc_datetime()

                tree.addi(start, end, set([job]))
                start = None
                end = None
        else:
            start = None
            end = None

tree.merge_overlaps(data_reducer=lambda x, y: x.union(y))
print(render_tree_str(tree))
print(len(tree))

# Create a directed graph
G = nx.DiGraph()

# Edges from source to each job
source_edges = [('source', job, 1) for job in jobs]

# Edges from jobs to timeslots
job_to_timeslot_edges = []
for interval in tree:
    for job in interval.data:
        job_to_timeslot_edges.append((job, interval, 1))

# Edges from source to each job
sink_edges = [(interval, 'sink', 1) for interval in tree]

edges = source_edges + job_to_timeslot_edges + sink_edges

# Add edges with capacities to the graph
for u, v, capacity in edges:
    G.add_edge(u, v, capacity=capacity)

# Define source and sink nodes
source = 'source'
sink = 'sink'

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

pos = nx.multipartite_layout(
    G,
    subset_key={
        'a': ['source'],
        'b': [job for job in jobs],
        'c': [interval for interval in tree],
        'd': ['sink']
    }
)
nx.draw_networkx_nodes(G, pos, nodelist=[job for job in jobs], cmap=plt.get_cmap('jet'), node_color='red', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=[interval for interval in tree], cmap=plt.get_cmap('jet'), node_color='blue', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=['source'], cmap=plt.get_cmap('jet'), node_color='green', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=['sink'], cmap=plt.get_cmap('jet'), node_color='green', node_size = 5)
nx.draw_networkx_labels(G, pos, labels={'source': 'source', 'sink': 'sink'})
nx.draw_networkx_edges(G, pos, edgelist=source_edges, edge_color='red', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=job_to_timeslot_edges, edge_color='blue', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=sink_edges, edge_color='green', arrows=False)
plt.show()

new_source_edges = []
new_job_to_timeslot_edges = []
new_sink_edges = []

for u in flow_dict:
    for v, flow in flow_dict[u].items():
        if isinstance(u, str) and u == 'source':
            if flow > 0:
                new_source_edges.append((u,v,1))
        elif isinstance(u, Job) and u in jobs:
            if flow > 0:
                new_job_to_timeslot_edges.append((u,v,1))
        elif isinstance(u, Interval) and u in tree:
            if flow > 0:
                new_sink_edges.append((u,v,1))
        print(f"Edge {u} -> {v} has flow {flow}")

pos = nx.multipartite_layout(
    G,
    subset_key={
        'a': ['source'],
        'b': [job for job in jobs],
        'c': [interval for interval in tree],
        'd': ['sink']
    }
)
nx.draw_networkx_nodes(G, pos, nodelist=[job for job in jobs], cmap=plt.get_cmap('jet'), node_color='red', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=[interval for interval in tree], cmap=plt.get_cmap('jet'), node_color='blue', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=['source'], cmap=plt.get_cmap('jet'), node_color='green', node_size = 5)
nx.draw_networkx_nodes(G, pos, nodelist=['sink'], cmap=plt.get_cmap('jet'), node_color='green', node_size = 5)
nx.draw_networkx_labels(G, pos, labels={'source': 'source', 'sink': 'sink'})
nx.draw_networkx_edges(G, pos, edgelist=new_source_edges, edge_color='red', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=new_job_to_timeslot_edges, edge_color='blue', arrows=False)
nx.draw_networkx_edges(G, pos, edgelist=new_sink_edges, edge_color='green', arrows=False)

plt.show()
