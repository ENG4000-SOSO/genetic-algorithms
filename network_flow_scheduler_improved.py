import os
from pathlib import Path
import json
from datetime import timezone
from typing import List, Dict, Tuple
from pprint import pp
from job import Job
from skyfield.api import wgs84, load, EarthSatellite, Time
from intervaltree import IntervalTree
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time


eph = load('de421.bsp')

ts = load.timescale()

altitude_degrees = 30.0

event_names = f'rise above {altitude_degrees}°', 'culminate', f'set below {altitude_degrees}°'


def counter_generator():
    i = 0
    while True:
        yield i
        i += 1


def parse_jobs(order_data_dir: Path) -> List[Job]:
    jobs = []
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

    return jobs


def parse_satellites(satellite_data_dir: Path) -> List[EarthSatellite]:
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

    return satellites


def update_trees_with_jobs(trees: Dict[EarthSatellite, IntervalTree], sat: EarthSatellite, job: Job, t0: Time, t1: Time) -> None:
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
    start = None

    for ti, event, sunlit_flag in zip(t, events, sunlit):
        if sunlit_flag:
            if event == 0:
                start = ti.utc_datetime()
            elif event == 2:
                if not start:
                    continue
                end = ti.utc_datetime()
                trees[sat].addi(start, end, set([job]))
                start = None
                end = None
        else:
            start = None
            end = None

def generate_trees(satellites: List[EarthSatellite], jobs: List[Job], t0: Time, t1: Time) -> Dict[EarthSatellite, IntervalTree]:
    # This is an interval tree.
    #
    # Each node of the tree stores an interval, which has:
    #   - start time
    #   - end time
    #   - data (containing the job that is able to be scheduled in this interval)
    trees = {sat: IntervalTree() for sat in satellites}

    for sat in satellites:
        for job in jobs:
            update_trees_with_jobs(trees, sat, job, t0, t1)

    # Merge overlaps in each interval tree.
    #
    # Whenever two intervals overlap, we create a new interval for them that
    # accommodates them both, and the jobs that can be scheduled in that interval is
    # the union of the jobs that could have been scheduled in the original
    # intervals.
    for sat, tree in trees.items():
        tree.merge_overlaps(data_reducer=lambda x, y: x.union(y))

    return trees


def generate_initial_graph_edges(
        trees: Dict[EarthSatellite, IntervalTree],
        jobs: List[Job],
        satellites: List[EarthSatellite]
    ) -> Tuple[
            List[Tuple[str, str, int]],
            Dict[EarthSatellite, List[Tuple[str, str, int]]],
            Dict[EarthSatellite, List[Tuple[str, str, int]]]
        ]:
    # Edges from source to each job
    source_edges = [('source', job.name, 1) for job in jobs]

    # Edges from jobs to timeslots.
    #
    # Remember: for each interval in the interval tree, we held the start and
    # end time of the interval, along with all the jobs that could be scheduled
    # in that interval.
    #
    # This for loop just creates edges between each job and the intervals that
    # it could be scheduled in.
    job_to_timeslot_edges = []
    sat_edges = {sat: [] for sat in satellites}
    for sat, tree in trees.items():
        for interval in tree:
            for job in interval.data:
                sat_edges[sat].append((job.name, f'{sat.name} {interval.begin} {interval.end}', 1))
                job_to_timeslot_edges.append((job.name, f'{sat.name} {interval.begin} {interval.end}', 1))

    # Edges from source to each job
    sink_edges = []
    sat_to_sink_edges = {sat: [] for sat in satellites}

    for sat, tree in trees.items():
        for interval in tree:
            sat_to_sink_edges[sat].append((f'{sat.name} {interval.begin} {interval.end}', 'sink', 1))

    return source_edges, sat_edges, sat_to_sink_edges


def extract_optimal_edges(
        flow_dict: dict,
        jobs: List[Job],
        satellites: List[EarthSatellite]
    ) -> Tuple[
            List[Tuple[str, str, int]],
            Dict[EarthSatellite, List[Tuple[str, str, int]]],
            Dict[EarthSatellite, List[Tuple[str, str, int]]]
        ]:
    new_source_edges = []
    new_sat_edges = {sat: [] for sat in satellites}
    new_sat_to_sink_edges = {sat: [] for sat in satellites}

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

    return new_source_edges, new_sat_edges, new_sat_to_sink_edges


def plot(
        G: nx.DiGraph,
        trees: Dict[EarthSatellite, IntervalTree],
        jobs: List[Job],
        satellites: List[EarthSatellite],
        source_edges: List[Tuple[str, str, int]],
        sat_edges: Dict[EarthSatellite, List[Tuple[str, str, int]]],
        sat_to_sink_edges: Dict[EarthSatellite, List[Tuple[str, str, int]]],
        title: str
    ) -> None:

    fig, ax = plt.subplots()

    pos = nx.multipartite_layout(
        G,
        subset_key={
            'a': ['source'],
            'b': [job.name for job in jobs],
            'c': [f'{sat.name} {interval.begin} {interval.end}' for sat, tree in trees.items() for interval in tree],
            'd': ['sink']
        }
    )

    def array_op(x):
        coords = pos[x]
        if x.startswith('Job'):
            return np.array((coords[0], coords[1]*5))
        elif x.startswith('SOSO'):
            return np.array((coords[0], coords[1]*2))
        else:
            return np.array((coords[0], coords[1]))

    pos = {p:array_op(p) for p in pos}

    source_color = 'white'
    jobs_color = 'black'
    soso1_color = '#4d4dff'
    soso2_color = 'green'
    soso3_color = '#FFEA00'
    soso4_color = 'orange'
    soso5_color = 'red'
    sink_color = 'black'

    nx.draw_networkx_nodes(G, pos, nodelist=[job.name for job in jobs], node_color=jobs_color, node_size = 5)

    nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval.begin} {interval.end}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-1'], node_color=soso1_color, node_size = 5, edgecolors='black', linewidths=0.1)
    nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval.begin} {interval.end}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-2'], node_color=soso2_color, node_size = 5, edgecolors='black', linewidths=0.1)
    nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval.begin} {interval.end}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-3'], node_color=soso3_color, node_size = 5, edgecolors='black', linewidths=0.1)
    nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval.begin} {interval.end}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-4'], node_color=soso4_color, node_size = 5, edgecolors='black', linewidths=0.1)
    nx.draw_networkx_nodes(G, pos, nodelist=[f'{sat.name} {interval.begin} {interval.end}' for sat, tree in trees.items() for interval in tree if sat.name == 'SOSO-5'], node_color=soso5_color, node_size = 5, edgecolors='black', linewidths=0.1)

    nx.draw_networkx_nodes(G, pos, nodelist=['source'], node_color=source_color, node_size = 15, edgecolors='black', linewidths=0.75)
    nx.draw_networkx_nodes(G, pos, nodelist=['sink'], node_color=sink_color, node_size = 15)

    nx.draw_networkx_edges(G, pos, edgelist=source_edges, edge_color=jobs_color, arrows=False)

    nx.draw_networkx_edges(G, pos, edgelist=sat_edges[satellites[0]], edge_color=soso1_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=sat_edges[satellites[1]], edge_color=soso2_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=sat_edges[satellites[2]], edge_color=soso3_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=sat_edges[satellites[3]], edge_color=soso4_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=sat_edges[satellites[4]], edge_color=soso5_color, arrows=False)

    nx.draw_networkx_edges(G, pos, edgelist=sat_to_sink_edges[satellites[0]], edge_color=soso1_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=sat_to_sink_edges[satellites[1]], edge_color=soso2_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=sat_to_sink_edges[satellites[2]], edge_color=soso3_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=sat_to_sink_edges[satellites[3]], edge_color=soso4_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=sat_to_sink_edges[satellites[4]], edge_color=soso5_color, arrows=False)

    plt.title(title)
    # Create a custom legend
    legend_elements_1 = [
        plt.Line2D([0], [0], marker='o', color=jobs_color, lw=0, markersize=5, label='Satellite Jobs')
    ]
    legend_elements_2 = [
        plt.Line2D([0], [0], marker='o', color=soso1_color, lw=0, markersize=5, label='Satellite 1 Timeslots'),
        plt.Line2D([0], [0], marker='o', color=soso2_color, lw=0, markersize=5, label='Satellite 2 Timeslots'),
        plt.Line2D([0], [0], marker='o', color=soso3_color, lw=0, markersize=5, label='Satellite 3 Timeslots'),
        plt.Line2D([0], [0], marker='o', color=soso4_color, lw=0, markersize=5, label='Satellite 4 Timeslots'),
        plt.Line2D([0], [0], marker='o', color=soso5_color, lw=0, markersize=5, label='Satellite 5 Timeslots')
    ]

    plt.text(0.01, 0.495, 'source', transform=ax.transAxes)
    plt.text(0.95, 0.495, 'sink', transform=ax.transAxes)
    plt.text(0.01, 0.02, 'Flow Source\n(Conceptual)', transform=ax.transAxes)
    plt.text(0.34, 0.02, 'Jobs', transform=ax.transAxes)
    plt.text(0.62, 0.02, 'Satellite\nTime Slots', transform=ax.transAxes)
    plt.text(0.9, 0.02, 'Flow Sink\n(Conceptual)', transform=ax.transAxes)

    # Add the legend to the axis
    legend_1 = ax.legend(handles=legend_elements_1, loc='upper left')
    ax.legend(handles=legend_elements_2, loc='upper right')
    ax.add_artist(legend_1)

    plt.show()


def main():
    parse_t0 = time.time()

    # === GET SATELLITES AND ORDERS DATA ===
    # Define directories containing the satellites and orders data
    project_dir = Path(os.path.dirname(__file__))
    data_dir = project_dir / 'data'
    order_data_dir = data_dir / 'orders'
    satellite_data_dir = data_dir / 'satellites'

    # Parse satellites and orders data
    satellites = parse_satellites(satellite_data_dir)
    jobs = parse_jobs(order_data_dir)

    parse_t1 = time.time()
    print(f'Parsing data took {parse_t1 - parse_t0} seconds')

    tree_t0 = time.time()

    # === DEFINE START AND END TIMES ===
    # The scheduling will cover the start of the earliest job to the end of the
    # latest job
    t0 = ts.from_datetime(min(job.start for job in jobs).replace(tzinfo=timezone.utc))
    t1 = ts.from_datetime(max(job.end for job in jobs).replace(tzinfo=timezone.utc))

    # === GENERATE INTERVAL TREES ===
    # One interval tree per satellite, where each interval in the tree holds the
    # jobs that could be scheduled in that interval
    trees = generate_trees(satellites, jobs, t0, t1)

    tree_t1 = time.time()
    print(f'Making the interval tree took {tree_t1 - tree_t0} seconds')

    graph_t0 = time.time()

    # === MODELING PROBLEM AS GRAPH AND GETTING MAXIMUM FLOW ===
    # Create an empty directed graph
    G = nx.DiGraph()

    # Generate edges for the graph
    source_to_jobs_edges, jobs_to_sat_edges, sat_to_sink_edges = \
        generate_initial_graph_edges(trees, jobs, satellites)

    sink_edges = [sat_to_sink_edge for sat_edges in sat_to_sink_edges.values() for sat_to_sink_edge in sat_edges]
    job_to_timeslot_edges = [job_to_timeslot_edge for sat_edges in jobs_to_sat_edges.values() for job_to_timeslot_edge in sat_edges]
    edges = source_to_jobs_edges + job_to_timeslot_edges + sink_edges

    # Add edges with capacities to the graph
    for u, v, capacity in edges:
        G.add_edge(u, v, capacity=capacity)

    # Define source and sink nodes
    source = 'source'
    sink = 'sink'

    # Calculate the maximum flow and the flow on each edge
    flow_value, flow_dict = nx.maximum_flow(G, source, sink)

    graph_t1 = time.time()
    print(f'Finding the maximum flow took {graph_t1 - graph_t0} seconds')

    plot(
        G,
        trees,
        jobs,
        satellites,
        source_to_jobs_edges,
        jobs_to_sat_edges,
        sat_to_sink_edges,
        'All Possible Scheduling Opportunities'
    )

    # Get the optimal edges from the graph
    new_source_edges, new_sat_edges, new_sat_to_sink_edges = \
        extract_optimal_edges(flow_dict, jobs, satellites)

    plot(
        G,
        trees,
        jobs,
        satellites,
        new_source_edges,
        new_sat_edges,
        new_sat_to_sink_edges,
        'Maximum Flow Optimal Schedule'
    )

    print(f'Total runtime (without plotting): {sum([parse_t1-parse_t0, tree_t1-tree_t0, graph_t1-graph_t0])}')


if __name__ == '__main__':
    main()
