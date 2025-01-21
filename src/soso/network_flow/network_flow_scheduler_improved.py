'''
Main functionality for the network flow optimization solution to the
job-satellite scheduling problem.
'''


import logging
import logging.config
from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from skyfield.api import EarthSatellite

from soso.debug import debug
from soso.interval_tree import GroundStationPassInterval, SatelliteInterval
from soso.job import Job
from soso.network_flow.edge_types import \
    Edges, \
    GroundStationPassTimeSlot, \
    GroundStationPassToSinkEdge, \
    JobToSatelliteTimeSlotEdge, \
    SatelliteTimeSlot, \
    SatelliteTimeSlotToGroundStationPassEdge, \
    SourceToJobEdge
from soso.network_flow.plot import plot


GROUND_STATION_PASS_CAPACITY = 10
'''
Number of jobs that can be downlinked at once during a ground station pass.

Note: this is an approximation and should be made more dynamic (by considering
uplink/downlink rates and payload sizes).
'''

ALTITUDE_DEGREES = 30.0
'''
Threshold for the degrees above the horizontal for a satellite to be considered
to have a point on Earth in its field-of-view.
'''

logger: logging.Logger = logging.getLogger(__name__)


def get_scheduled_and_unscheduled_jobs(
    jobs: List[Job],
    jobs_to_sat_edges: Dict[EarthSatellite, List[JobToSatelliteTimeSlotEdge]]
) -> Tuple[Set[Job], Set[Job]]:
    '''
    Gets scheduled and unscheduled jobs given edges in a network flow graph.

    A job is considered scheduled if it has an edge to a satellite timeslot.
    Otherwise, it is not scheduled.

    Note that this function can also be used to find schedulable and
    unschedulable jobs (if that's what the underlying network flow graph represents).

    Args:
        jobs: List of all jobs.

        jobs_to_sat_edges: Edges in the network flow graph from jobs to
        satellite timeslots.

    Returns:
        A tuple containing two sets, the first being the scheduled jobs and the
        second being the unscheduled jobs.
    '''

    jobs_set = set(jobs)

    schedulable_jobs: Set[Job] = set()
    for sat_edges in jobs_to_sat_edges.values():
        for job_to_timeslot_edge in sat_edges:
            schedulable_jobs.add(job_to_timeslot_edge.job)

    unschedulable_jobs = jobs_set.difference(schedulable_jobs)

    return schedulable_jobs, unschedulable_jobs


@debug
def debug_network_info(
    G: nx.DiGraph,
    unoptimized_edges: Edges,
    optimal_edges: Edges,
    satellite_intervals: Dict[EarthSatellite, List[SatelliteInterval]],
    ground_station_passes: Dict[EarthSatellite, List[GroundStationPassInterval]],
    satellites: List[EarthSatellite],
    jobs: List[Job],
    debug_mode: Optional[Path | bool] = None
) -> None:
    '''
    Logs debug information, both as logger messages and images.

    Don't worry too much about this function. It's just for debugging.

    Args:
        G: The network flow graph.

        unoptimized_edges: Edges before optimized by the network flow algorithm.

        optimal_edges: Edges before optimized by the network flow algorithm.

        satellite_intervals: Dictionary mapping satellites to intervals.

        ground_station_passes: Dictionary mapping satellites to ground station
        passes.

        satellites: The list of satellites.

        jobs: The list of jobs,

        debug_mode: The debug mode. See the `run_network_flow` function.
    '''

    # Unpack unoptimized edges
    source_to_jobs_edges = unoptimized_edges.sourceToJobEdges
    jobs_to_sat_edges = unoptimized_edges.jobToSatelliteTimeSlotEdges
    sat_to_ground_pass_edges = unoptimized_edges.satelliteTimeSlotToGroundStationPassEdge
    ground_pass_to_sink_edges = unoptimized_edges.groundStationPassToSinkEdge

    schedulable_jobs, unschedulable_jobs = \
        get_scheduled_and_unscheduled_jobs(jobs, jobs_to_sat_edges)

    logger.debug(f'{len(unschedulable_jobs)} not schedulable')

    for job in unschedulable_jobs:
        logger.debug(f'{job} not schedulable')

    # Unpack optimal edges
    new_source_edges = optimal_edges.sourceToJobEdges
    new_jobs_to_sat_edges = optimal_edges.jobToSatelliteTimeSlotEdges
    new_sat_to_ground_pass_edges = optimal_edges.satelliteTimeSlotToGroundStationPassEdge
    new_ground_pass_to_sink_edges = optimal_edges.groundStationPassToSinkEdge

    scheduled_jobs, unscheduled_jobs = \
        get_scheduled_and_unscheduled_jobs(jobs, new_jobs_to_sat_edges)

    logger.debug(f'{len(unscheduled_jobs)} not scheduled')

    for job in unscheduled_jobs:
        logger.debug(f'{job} not scheduled')

    logger.info(
        f'Out of {len(jobs)} jobs, {len(scheduled_jobs)} were '
            f'scheduled, {len(unscheduled_jobs)} were not, '
            f'({len(unschedulable_jobs)} jobs were not possible to be '
            'scheduled)'
    )

    # Plot all possible scheduling opportunities as network flow graph
    plot(
        G,
        satellite_intervals,
        ground_station_passes,
        jobs,
        satellites,
        source_to_jobs_edges,
        jobs_to_sat_edges,
        sat_to_ground_pass_edges,
        ground_pass_to_sink_edges,
        'All Possible Scheduling Opportunities',
        debug_mode
    )

    # Plot optimal schedule as network flow graph
    plot(
        G,
        satellite_intervals,
        ground_station_passes,
        jobs,
        satellites,
        new_source_edges,
        new_jobs_to_sat_edges,
        new_sat_to_ground_pass_edges,
        new_ground_pass_to_sink_edges,
        'Maximum Flow Optimal Schedule',
        debug_mode
    )


def generate_initial_graph_edges(
    satellite_intervals: Dict[EarthSatellite, List[SatelliteInterval]],
    ground_station_passes: Dict[EarthSatellite, List[GroundStationPassInterval]],
    jobs: List[Job],
    satellites: List[EarthSatellite]
) -> Edges:
    '''
    Generates initial graph edges for a network flow given satellites, jobs, and
    interval trees.

    The edges generated are:
        - an edge from the source node to each job,
        - 0 or more edges from each job to the satellite timeslots that it can
          be scheduled into, and
        - one edge from each satellite timeslot to the sink node.

    Args:
        satellite_intervals: The dictionary mapping each satellite to its list
        of schedulable intervals.

        ground_station_passes: The dictionary mapping each satellite to its list
        of ground station passes.

        jobs: The full list of jobs.

        satellites: The full list of satellites.

    Returns:
        A representation of the edges in the graph.
    '''

    # Edges from source to each job
    source_edges = [SourceToJobEdge('source', job, 1) for job in jobs]

    # Edges from jobs to timeslots.
    #
    # Remember: for each interval in the interval tree, we held the start and
    # end time of the interval, along with all the jobs that could be scheduled
    # in that interval.
    #
    # This for loop just creates edges between each job and the intervals that
    # it could be scheduled in.
    sat_edges: Dict[EarthSatellite, List[JobToSatelliteTimeSlotEdge]] = {
        sat: [] for sat in satellites
    }
    for sat, intervals in satellite_intervals.items():
        for interval in intervals:
            for job in interval.data:
                sat_edges[sat].append(
                    JobToSatelliteTimeSlotEdge(
                        job,
                        SatelliteTimeSlot(sat, interval.begin, interval.end),
                        1
                    )
                )

    # Edges from satellite job timeslot to ground station pass timeslots
    sat_to_ground_pass_edges: Dict[EarthSatellite, List[SatelliteTimeSlotToGroundStationPassEdge]] = {
        sat: [] for sat in satellites
    }
    for sat, intervals in satellite_intervals.items():
        for interval in intervals:
            for ground_station_pass in ground_station_passes[sat]:
                if ground_station_pass.begin > interval.end:
                    sat_to_ground_pass_edges[sat].append(
                        SatelliteTimeSlotToGroundStationPassEdge(
                            SatelliteTimeSlot(sat, interval.begin, interval.end),
                            GroundStationPassTimeSlot(
                                sat,
                                ground_station_pass.ground_station,
                                ground_station_pass.begin,
                                ground_station_pass.end
                            ),
                            1
                        )
                    )

    # Edges from ground station pass timeslots to the sink
    ground_pass_to_sink_edges: Dict[EarthSatellite, List[GroundStationPassToSinkEdge]] = {
        sat: [] for sat in satellites
    }
    for sat, intervals in ground_station_passes.items():
        for interval in intervals:
            ground_pass_to_sink_edges[sat].append(
                GroundStationPassToSinkEdge(
                    GroundStationPassTimeSlot(
                        sat,
                        interval.ground_station,
                        interval.begin,
                        interval.end
                    ),
                    'sink',
                    GROUND_STATION_PASS_CAPACITY
                )
            )

    return Edges(
        source_edges,
        sat_edges,
        sat_to_ground_pass_edges,
        ground_pass_to_sink_edges
    )


def extract_optimal_edges(
    flow_dict: dict,
    satellites: List[EarthSatellite]
) -> Edges:
    '''
    Extracts the edges from an optimal flow network. Only the edges with nonzero
    flow are extracted.

    Args:
        flow_dict: A `dict` containing the value of the flow through each edge
            in the graph. This should be returned from the `maximum_flow`
            function of the `networkx` library.

        satellites: The complete list of satellites.

    Returns:
        A representation of the edges in the graph.
    '''

    new_source_edges: List[SourceToJobEdge] = []
    new_sat_edges: Dict[EarthSatellite, List[JobToSatelliteTimeSlotEdge]] = {
        sat: [] for sat in satellites
    }
    new_sat_to_ground_pass_edges: Dict[EarthSatellite, List[SatelliteTimeSlotToGroundStationPassEdge]] = {
        sat: [] for sat in satellites
    }
    new_ground_pass_to_sink_edges: Dict[EarthSatellite, List[GroundStationPassToSinkEdge]] = {
        sat: [] for sat in satellites
    }

    for u in flow_dict:
        for v, flow in flow_dict[u].items():
            if isinstance(u, str) and u == 'source':
                if flow > 0:
                    new_source_edges.append(SourceToJobEdge(u, v, flow))
            elif isinstance(u, Job):
                if flow > 0:
                    if isinstance(v, SatelliteTimeSlot):
                        new_sat_edges[v.satellite].append(
                            JobToSatelliteTimeSlotEdge(u, v, flow)
                        )
            elif isinstance(u, SatelliteTimeSlot):
                if flow > 0:
                    new_sat_to_ground_pass_edges[u.satellite].append(
                        SatelliteTimeSlotToGroundStationPassEdge(u, v, flow)
                    )
            elif isinstance(u, GroundStationPassTimeSlot):
                if flow > 0:
                    new_ground_pass_to_sink_edges[u.satellite].append(
                        GroundStationPassToSinkEdge(u, v, flow)
                    )

    return Edges(
        new_source_edges,
        new_sat_edges,
        new_sat_to_ground_pass_edges,
        new_ground_pass_to_sink_edges
    )


def run_network_flow(
    satellites: List[EarthSatellite],
    jobs: List[Job],
    satellite_intervals: Dict[EarthSatellite, List[SatelliteInterval]],
    ground_station_passes: Dict[EarthSatellite, List[GroundStationPassInterval]],
    debug_mode: Optional[Path | bool] = None
) -> Dict[EarthSatellite, List[JobToSatelliteTimeSlotEdge]]:
    '''
    The main entry point to the network flow part of the SOSO scheduling
    algorithm.

    Args:
        satellites: The list of satellites to be scheduled with jobs and outage
        requests.

        jobs: The list of jobs to be scheduled.

        satellite_intervals: The dictionary mapping each satellite to its list
        of schedulable intervals.

        ground_station_passes: The dictionary mapping each satellite to its list
        of ground station passes.

        debug_mode: Information about how debugging images should be
        handled. If it is set to a `Path` object, images will be outputted in
        that directory. If it is `True`, images will be displayed during the
        execution. Otherwise, if it is `None`, no debugging images will be shown
        or saved at all.

    Returns:
        A dictionary mapping each satellite to a list, where items in the list
        are representations of a job scheduled in a time slot.
    '''

    logger.info('Starting network flow algorithm')

    alg_t0 = time.time()

    graph_t0 = time.time()

    # Create an empty directed graph
    G = nx.DiGraph()

    # Generate edges for the graph
    edges = generate_initial_graph_edges(
        satellite_intervals,
        ground_station_passes,
        jobs,
        satellites
    )

    # Unpack edges
    source_to_jobs_edges = edges.sourceToJobEdges
    jobs_to_sat_edges = edges.jobToSatelliteTimeSlotEdges
    sat_to_ground_pass_edges = edges.satelliteTimeSlotToGroundStationPassEdge
    ground_pass_to_sink_edges = edges.groundStationPassToSinkEdge

    # Convert job-to-timeslot edges from a dictionary to a list
    job_to_timeslot_edges = [
        edge for edges in jobs_to_sat_edges.values() for edge in edges
    ]

    # Convert timeslot-to-ground-station-pass edges from a dictionary to a list
    timeslot_to_ground_station_edges = [
        edge for edges in sat_to_ground_pass_edges.values() for edge in edges
    ]

    # Convert ground-station-pass-to-sink edges from a dictionary to a list
    sink_edges = [
        edge for edges in ground_pass_to_sink_edges.values() for edge in edges
    ]

    # Combine all edges into one big edge list
    graph_edges = source_to_jobs_edges + job_to_timeslot_edges + timeslot_to_ground_station_edges + sink_edges

    # Add edges with capacities to the graph
    for u, v, capacity in graph_edges:
        G.add_edge(u, v, capacity=capacity)

    # Define source and sink nodes
    source = 'source'
    sink = 'sink'

    try:
        # Calculate the maximum flow and the flow on each edge
        flow_value, flow_dict = nx.maximum_flow(G, source, sink)
    except nx.exception.NetworkXError as e:
        # Sometimes we get the exception that the sink node is not in the graph.
        # In this scenario, it means that there are no edges from the any of the
        # satellite time slots to the sink, so no schedule is possible. We
        # handle that error gracefully here.
        logger.error(f'networkx exception: returning early')
        logger.error(f'networkx exception: {e}')
        return {sat: [] for sat in satellites}

    graph_t1 = time.time()
    logger.debug(f'Finding the maximum flow took {graph_t1 - graph_t0} seconds')

    # Get the optimal edges from the graph
    optimal_edges = extract_optimal_edges(flow_dict, satellites)

    new_sat_edges = optimal_edges.jobToSatelliteTimeSlotEdges

    alg_t1 = time.time()

    logger.info(f'Total runtime (without plotting): {alg_t1 - alg_t0}')

    debug_network_info(
        G,
        edges,
        optimal_edges,
        satellite_intervals,
        ground_station_passes,
        satellites,
        jobs,
        debug_mode
    )

    return new_sat_edges
