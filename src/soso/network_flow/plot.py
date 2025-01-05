from typing import Dict, List
from intervaltree import IntervalTree
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skyfield.api import EarthSatellite
from soso.debug import debug
from soso.job import Job
from soso.network_flow.edge_types import SourceToJobEdge, JobToSatelliteTimeSlotEdge, SatelliteTimeSlotToSinkEdge


@debug
def plot(
        G: nx.DiGraph,
        trees: Dict[EarthSatellite, IntervalTree],
        jobs: List[Job],
        satellites: List[EarthSatellite],
        source_edges: List[SourceToJobEdge],
        sat_edges: Dict[EarthSatellite, List[JobToSatelliteTimeSlotEdge]],
        sat_to_sink_edges: Dict[EarthSatellite, List[SatelliteTimeSlotToSinkEdge]],
        title: str
    ) -> None:
    '''
    Plots a flow network representing jobs scheduled into satellites.

    Args:
        G: The `networkx` graph of the flow network.

        trees: The `dict` of interval trees (each satellite's interval tree).

        jobs: The full list of jobs.

        satellites: The full list of satellites.

        source_edges: Edges in the graph from source to jobs. Tuples are of the
            form `(sink, job, flow)`.

        sat_edges: Edges in the graph from job to satellite timeslot for each
            satellite. Tuples are of the form `(job, satellite_timeslot, flow)`.

        sat_to_sink_edges: Edges in the graph from satellite timeslot to sink
            for each satellite. Tuples are of the form
            `(satellite_timeslot, sink, flow)`.
    '''

    fig, ax = plt.subplots()

    pos = nx.multipartite_layout(
        G,
        subset_key={
            'a': ['source'],
            'b': [job.name for job in jobs],
            'c': [
                f'{sat.name} {interval.begin} {interval.end}'
                for sat, tree in trees.items() for interval in tree
            ],
            'd': ['sink']
        }
    )

    def array_op(x):
        coords = pos[x]
        if x.startswith('Job'):
            return np.array((coords[0], coords[1]*5))
        elif x.startswith('SOSO'):
            return np.array((coords[0], coords[1]*1.2))
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
