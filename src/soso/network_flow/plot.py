# type: ignore
#
# This comment disables type checking for this module. It's too much effort to
# fix the type hints for a module that's only used for debugging :P.

'''
Functionality related to plotting network flow graphs.

This gives us a visual representation of the network flow graph that models the
SOSO optimization problem, which is useful for debugging.
'''


import datetime
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skyfield.api import EarthSatellite

from soso.debug import debug
from soso.interval_tree import GroundStationPassInterval, SatelliteInterval
from soso.job import Job
from soso.network_flow import \
    GroundStationPassTimeSlot, \
    GroundStationPassToSinkEdge, \
    JobToSatelliteTimeSlotEdge, \
    RateLimiter, \
    RateLimiterEdge, \
    SatelliteTimeSlot, \
    SatelliteTimeSlotToRateLimiter, \
    SatelliteToList, \
    SourceToJobEdge


@debug
def plot(
        G: nx.DiGraph,
        satellite_intervals: SatelliteToList[SatelliteInterval],
        ground_station_passes: SatelliteToList[GroundStationPassInterval],
        jobs: List[Job],
        satellites: List[EarthSatellite],
        source_edges: List[SourceToJobEdge],
        job_to_sat_edges: SatelliteToList[JobToSatelliteTimeSlotEdge],
        sat_to_rate_limiter_edges: SatelliteToList[SatelliteTimeSlotToRateLimiter],
        rate_limiter_to_ground_station: SatelliteToList[RateLimiterEdge],
        ground_station_to_sink_edges: SatelliteToList[GroundStationPassToSinkEdge],
        title: str,
        debug_mode: Optional[Path | bool] = None    
    ) -> None:
    '''
    Plots a flow network representing jobs scheduled into satellites.

    Args:
        G: The `networkx` graph of the flow network.

        satellite_intervals: The dictionary of satellite's intervals (each
        satellite's list of intervals).

        ground_station_passes: The dictionary of each satellite's ground station
        passes.

        jobs: The full list of jobs.

        satellites: The full list of satellites.

        source_edges: Edges in the graph from source to jobs.

        job_to_sat_edges: Edges in the graph from job to satellite timeslot for
        each satellite.

        sat_to_ground_station_edges: Edges in the graph from satellite timeslot
        to ground station pass.

        ground_station_to_sink_edges: Edges in the graph from ground station to
        sink.

        title: The title of the plot.

        debug_mode: The debug mode. Either a `Path` (which will place output
        images in that directory), `True`, which will display output images to
        the user, or `None`, which will not show or save images at all.
    '''

    # Disable debug messages from the Matplotlib logger
    plt.set_loglevel(level = 'warning')

    # Disable debug messages from the Pillow logger
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    fig, ax = plt.subplots()

    pos = nx.multipartite_layout(
        G,
        subset_key={
            'a': ['source'],
            'b': jobs,
            'c': [
                SatelliteTimeSlot(sat, interval.begin, interval.end)
                    for sat, intervals in satellite_intervals.items()
                        for interval in intervals
            ],
            'd': [
                RateLimiter(SatelliteTimeSlot(sat, interval.begin, interval.end))
                    for sat, intervals in satellite_intervals.items()
                        for interval in intervals
            ],
            'e': [
                GroundStationPassTimeSlot(
                    sat,
                    interval.ground_station,
                    interval.begin,
                    interval.end
                )
                    for sat, intervals in ground_station_passes.items()
                        for interval in intervals
            ],
            'f': ['sink']
        }
    )


    def array_op(x):
        coords = pos[x]
        if isinstance(x, Job):
            return np.array((coords[0], coords[1]*5))
        elif isinstance(x, SatelliteTimeSlot):
            a = coords[0]
            return np.array((coords[0], coords[1]*1.2))
        elif isinstance(x, RateLimiter):
            a = coords[0]
            return np.array((coords[0], coords[1]*1.2))
        elif isinstance(x, GroundStationPassTimeSlot):
            return np.array((coords[0], coords[1]*10))
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

    nx.draw_networkx_nodes(G, pos, nodelist=[job for job in jobs], node_color=jobs_color, node_size = 5)

    nx.draw_networkx_nodes(G, pos, nodelist=[SatelliteTimeSlot(sat, interval.begin, interval.end) for sat, interval in satellite_intervals.items() for interval in interval if sat.name == 'SOSO-1'], node_color=soso1_color, node_size = 5, edgecolors='black', linewidths=0.1)
    nx.draw_networkx_nodes(G, pos, nodelist=[SatelliteTimeSlot(sat, interval.begin, interval.end) for sat, interval in satellite_intervals.items() for interval in interval if sat.name == 'SOSO-2'], node_color=soso2_color, node_size = 5, edgecolors='black', linewidths=0.1)
    nx.draw_networkx_nodes(G, pos, nodelist=[SatelliteTimeSlot(sat, interval.begin, interval.end) for sat, interval in satellite_intervals.items() for interval in interval if sat.name == 'SOSO-3'], node_color=soso3_color, node_size = 5, edgecolors='black', linewidths=0.1)
    nx.draw_networkx_nodes(G, pos, nodelist=[SatelliteTimeSlot(sat, interval.begin, interval.end) for sat, interval in satellite_intervals.items() for interval in interval if sat.name == 'SOSO-4'], node_color=soso4_color, node_size = 5, edgecolors='black', linewidths=0.1)
    nx.draw_networkx_nodes(G, pos, nodelist=[SatelliteTimeSlot(sat, interval.begin, interval.end) for sat, interval in satellite_intervals.items() for interval in interval if sat.name == 'SOSO-5'], node_color=soso5_color, node_size = 5, edgecolors='black', linewidths=0.1)

    nx.draw_networkx_nodes(G, pos, nodelist=[RateLimiter(SatelliteTimeSlot(sat, interval.begin, interval.end)) for sat, interval in satellite_intervals.items() for interval in interval if sat.name == 'SOSO-1'], node_color=soso1_color, node_size = 5, edgecolors='black', linewidths=0.1)
    nx.draw_networkx_nodes(G, pos, nodelist=[RateLimiter(SatelliteTimeSlot(sat, interval.begin, interval.end)) for sat, interval in satellite_intervals.items() for interval in interval if sat.name == 'SOSO-2'], node_color=soso2_color, node_size = 5, edgecolors='black', linewidths=0.1)
    nx.draw_networkx_nodes(G, pos, nodelist=[RateLimiter(SatelliteTimeSlot(sat, interval.begin, interval.end)) for sat, interval in satellite_intervals.items() for interval in interval if sat.name == 'SOSO-3'], node_color=soso3_color, node_size = 5, edgecolors='black', linewidths=0.1)
    nx.draw_networkx_nodes(G, pos, nodelist=[RateLimiter(SatelliteTimeSlot(sat, interval.begin, interval.end)) for sat, interval in satellite_intervals.items() for interval in interval if sat.name == 'SOSO-4'], node_color=soso4_color, node_size = 5, edgecolors='black', linewidths=0.1)
    nx.draw_networkx_nodes(G, pos, nodelist=[RateLimiter(SatelliteTimeSlot(sat, interval.begin, interval.end)) for sat, interval in satellite_intervals.items() for interval in interval if sat.name == 'SOSO-5'], node_color=soso5_color, node_size = 5, edgecolors='black', linewidths=0.1)

    nx.draw_networkx_nodes(G, pos, nodelist=[GroundStationPassTimeSlot(sat, interval.ground_station, interval.begin, interval.end) for sat, interval in ground_station_passes.items() for interval in interval if sat.name == 'SOSO-1'], edgecolors=soso1_color, node_size = 5, node_color='white', linewidths=1)
    nx.draw_networkx_nodes(G, pos, nodelist=[GroundStationPassTimeSlot(sat, interval.ground_station, interval.begin, interval.end) for sat, interval in ground_station_passes.items() for interval in interval if sat.name == 'SOSO-2'], edgecolors=soso2_color, node_size = 5, node_color='white', linewidths=1)
    nx.draw_networkx_nodes(G, pos, nodelist=[GroundStationPassTimeSlot(sat, interval.ground_station, interval.begin, interval.end) for sat, interval in ground_station_passes.items() for interval in interval if sat.name == 'SOSO-3'], edgecolors=soso3_color, node_size = 5, node_color='white', linewidths=1)
    nx.draw_networkx_nodes(G, pos, nodelist=[GroundStationPassTimeSlot(sat, interval.ground_station, interval.begin, interval.end) for sat, interval in ground_station_passes.items() for interval in interval if sat.name == 'SOSO-4'], edgecolors=soso4_color, node_size = 5, node_color='white', linewidths=1)
    nx.draw_networkx_nodes(G, pos, nodelist=[GroundStationPassTimeSlot(sat, interval.ground_station, interval.begin, interval.end) for sat, interval in ground_station_passes.items() for interval in interval if sat.name == 'SOSO-5'], edgecolors=soso5_color, node_size = 5, node_color='white', linewidths=1)

    nx.draw_networkx_nodes(G, pos, nodelist=['source'], node_color=source_color, node_size = 15, edgecolors='black', linewidths=0.75)
    nx.draw_networkx_nodes(G, pos, nodelist=['sink'], node_color=sink_color, node_size = 15)

    nx.draw_networkx_edges(G, pos, edgelist=source_edges, edge_color=jobs_color, arrows=False)

    nx.draw_networkx_edges(G, pos, edgelist=job_to_sat_edges[satellites[0]], edge_color=soso1_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=job_to_sat_edges[satellites[1]], edge_color=soso2_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=job_to_sat_edges[satellites[2]], edge_color=soso3_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=job_to_sat_edges[satellites[3]], edge_color=soso4_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=job_to_sat_edges[satellites[4]], edge_color=soso5_color, arrows=False)

    nx.draw_networkx_edges(G, pos, edgelist=sat_to_rate_limiter_edges[satellites[0]], edge_color=soso1_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=sat_to_rate_limiter_edges[satellites[1]], edge_color=soso2_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=sat_to_rate_limiter_edges[satellites[2]], edge_color=soso3_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=sat_to_rate_limiter_edges[satellites[3]], edge_color=soso4_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=sat_to_rate_limiter_edges[satellites[4]], edge_color=soso5_color, arrows=False)

    nx.draw_networkx_edges(G, pos, edgelist=rate_limiter_to_ground_station[satellites[0]], edge_color=soso1_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=rate_limiter_to_ground_station[satellites[1]], edge_color=soso2_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=rate_limiter_to_ground_station[satellites[2]], edge_color=soso3_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=rate_limiter_to_ground_station[satellites[3]], edge_color=soso4_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=rate_limiter_to_ground_station[satellites[4]], edge_color=soso5_color, arrows=False)

    nx.draw_networkx_edges(G, pos, edgelist=ground_station_to_sink_edges[satellites[0]], edge_color=soso1_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=ground_station_to_sink_edges[satellites[1]], edge_color=soso2_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=ground_station_to_sink_edges[satellites[2]], edge_color=soso3_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=ground_station_to_sink_edges[satellites[3]], edge_color=soso4_color, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=ground_station_to_sink_edges[satellites[4]], edge_color=soso5_color, arrows=False)

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
        plt.Line2D([0], [0], marker='o', color=soso5_color, lw=0, markersize=5, label='Satellite 5 Timeslots'),
        plt.Line2D([0], [0], marker='o', color=soso1_color, fillstyle='none', linewidth=1, lw=0, markersize=5, label='Satellite 1 Ground Station Passes'),
        plt.Line2D([0], [0], marker='o', color=soso2_color, fillstyle='none', linewidth=1, lw=0, markersize=5, label='Satellite 2 Ground Station Passes'),
        plt.Line2D([0], [0], marker='o', color=soso3_color, fillstyle='none', linewidth=1, lw=0, markersize=5, label='Satellite 3 Ground Station Passes'),
        plt.Line2D([0], [0], marker='o', color=soso4_color, fillstyle='none', linewidth=1, lw=0, markersize=5, label='Satellite 4 Ground Station Passes'),
        plt.Line2D([0], [0], marker='o', color=soso5_color, fillstyle='none', linewidth=1, lw=0, markersize=5, label='Satellite 5 Ground Station Passes')
    ]

    plt.text(0.01, 0.495, 'source', transform=ax.transAxes)
    plt.text(0.95, 0.495, 'sink', transform=ax.transAxes)
    plt.text(0.01, 0.02, 'Flow Source\n(Conceptual)', transform=ax.transAxes)
    plt.text(0.22, 0.02, 'Jobs', transform=ax.transAxes)
    plt.text(0.39, 0.02, 'Satellite Job\nTime Slots', transform=ax.transAxes)
    plt.text(0.56, 0.02, '1-1 Mapping for\nFlow Limiting', transform=ax.transAxes)
    plt.text(0.74, 0.02, 'Ground Station\nPass Time Slots', transform=ax.transAxes)
    plt.text(0.9, 0.02, 'Flow Sink\n(Conceptual)', transform=ax.transAxes)

    # Add the legend to the axis
    legend_1 = ax.legend(handles=legend_elements_1, loc='upper left')
    ax.legend(handles=legend_elements_2, loc='upper right')
    ax.add_artist(legend_1)

    if isinstance(debug_mode, Path):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        filename = f'{timestamp}_{title}.png'
        fig.set_size_inches(18, 12)
        plt.savefig(debug_mode / filename)
    elif debug_mode:
        plt.show()
