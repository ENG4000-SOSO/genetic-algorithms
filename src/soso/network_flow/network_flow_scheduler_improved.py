'''
Main functionality for the network flow optimization solution to the
job-satellite scheduling problem.
'''


import logging
import logging.config
import time
from typing import Dict, List, Set

import networkx as nx
from skyfield.api import EarthSatellite

from soso.interval_tree import SatelliteInterval
from soso.job import Job
from soso.network_flow.edge_types import \
    Edges, \
    JobToSatelliteTimeSlotEdge, \
    SatelliteTimeSlot, \
    SatelliteTimeSlotToSinkEdge, \
    SourceToJobEdge
from soso.network_flow.plot import plot


ALTITUDE_DEGREES = 30.0
'''
Threshold for the degrees above the horizontal for a satellite to be considered
to have a point on Earth in its field-of-view.
'''

logger: logging.Logger = logging.getLogger(__name__)


def generate_initial_graph_edges(
        satellite_intervals: Dict[EarthSatellite, List[SatelliteInterval]],
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

    # Edges from timeslots to the sink
    sat_to_sink_edges: Dict[EarthSatellite, List[SatelliteTimeSlotToSinkEdge]] = {
        sat: [] for sat in satellites
    }
    for sat, intervals in satellite_intervals.items():
        for interval in intervals:
            sat_to_sink_edges[sat].append(
                SatelliteTimeSlotToSinkEdge(
                    SatelliteTimeSlot(sat, interval.begin, interval.end),
                    'sink',
                    1
                )
            )

    return Edges(source_edges, sat_edges, sat_to_sink_edges)


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
    new_sat_to_sink_edges: Dict[EarthSatellite, List[SatelliteTimeSlotToSinkEdge]] = {
        sat: [] for sat in satellites
    }

    for u in flow_dict:
        for v, flow in flow_dict[u].items():
            if isinstance(u, str) and u == 'source':
                if flow > 0:
                    new_source_edges.append(SourceToJobEdge(u,v,1))
            elif isinstance(u, Job):
                if flow > 0:
                    if isinstance(v, SatelliteTimeSlot):
                        new_sat_edges[v.satellite].append(
                            JobToSatelliteTimeSlotEdge(u, v, 1)
                        )
            elif isinstance(u, SatelliteTimeSlot):
                if flow > 0:
                    new_sat_to_sink_edges[u.satellite].append(
                        SatelliteTimeSlotToSinkEdge(u, v, 1)
                    )

    return Edges(new_source_edges, new_sat_edges, new_sat_to_sink_edges)


def run_network_flow(
    satellites: List[EarthSatellite],
    jobs: List[Job],
    satellite_intervals: Dict[EarthSatellite, List[SatelliteInterval]]
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

    Returns:
        A dictionary mapping each satellite to a list, where items in the list
        are representations of a job scheduled in a time slot.
    '''

    logger.info('Starting network flow algorithm')

    alg_t0 = time.time()

    # Set of all jobs, will be used later
    all_jobs = set([job for job in jobs])

    graph_t0 = time.time()

    # Create an empty directed graph
    G = nx.DiGraph()

    # Generate edges for the graph
    edges = \
        generate_initial_graph_edges(satellite_intervals, jobs, satellites)

    # Unpack edges
    source_to_jobs_edges = edges.sourceToJobEdges
    jobs_to_sat_edges = edges.jobToSatelliteTimeSlotEdges
    sat_to_sink_edges = edges.satelliteTimeSlotToSinkEdges

    # Collect all jobs that are potentially schedulable
    schedulable_jobs: Set[Job] = set()
    for sat_edges in jobs_to_sat_edges.values():
        for job_to_timeslot_edge in sat_edges:
            schedulable_jobs.add(job_to_timeslot_edge.job)
    # Collect all jobs that are impossible to schedule
    unschedulable_jobs = all_jobs.difference(schedulable_jobs)
    for job_name in unschedulable_jobs:
        logger.debug(f'{job_name} not schedulable')

    # Collect job-to-timeslot edges and sink edges into lists (instead of dicts)
    job_to_timeslot_edges = [
        job_to_timeslot_edge
            for sat_edges in jobs_to_sat_edges.values()
                for job_to_timeslot_edge in sat_edges
    ]
    sink_edges = [
        sat_to_sink_edge
            for sat_edges in sat_to_sink_edges.values()
                for sat_to_sink_edge in sat_edges
    ]

    # Combine all edges into one big edge list
    graph_edges = source_to_jobs_edges + job_to_timeslot_edges + sink_edges

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
    optimal_edges = \
        extract_optimal_edges(flow_dict, satellites)

    # Unpack optimal edges
    new_source_edges = optimal_edges.sourceToJobEdges
    new_sat_edges = optimal_edges.jobToSatelliteTimeSlotEdges
    new_sat_to_sink_edges = optimal_edges.satelliteTimeSlotToSinkEdges

    # Collect all jobs that have been scheduled
    scheduled_jobs: Set[Job] = set()
    for sat_edges in new_sat_edges.values():
        for job_to_timeslot_edge in sat_edges:
            scheduled_jobs.add(job_to_timeslot_edge.job)
            logger.debug(
                f'{job_to_timeslot_edge.job} scheduled at '
                    f'{job_to_timeslot_edge.satellite_timeslot}'
            )

    # Collect all jobs that are impossible to schedule
    unscheduled_jobs = all_jobs.difference(scheduled_jobs)
    for job_name in unscheduled_jobs:
        logger.debug(f'{job_name} not scheduled')

    alg_t1 = time.time()

    logger.info(
        f'Out of {len(all_jobs)} jobs, {len(scheduled_jobs)} were '
            f'scheduled, {len(unscheduled_jobs)} were not, '
            f'({len(unschedulable_jobs)} jobs were not possible to be '
            'scheduled)'
    )
    logger.info(f'Total runtime (without plotting): {alg_t1 - alg_t0}')

    # Plot all possible scheduling opportunities as network flow graph
    plot(
        G,
        satellite_intervals,
        jobs,
        satellites,
        source_to_jobs_edges,
        jobs_to_sat_edges,
        sat_to_sink_edges,
        'All Possible Scheduling Opportunities'
    )

    # Plot optimal schedule as network flow graph
    plot(
        G,
        satellite_intervals,
        jobs,
        satellites,
        new_source_edges,
        new_sat_edges,
        new_sat_to_sink_edges,
        'Maximum Flow Optimal Schedule'
    )

    return new_sat_edges
