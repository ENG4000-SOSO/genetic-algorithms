'''
Main functionality for the network flow optimization solution to the
job-satellite scheduling problem.

The network flow algorithm is applied here to produce an optimal schedule. To do
this, we model the schedule as a directed graph, where an edge between two nodes
means one node can be scheduled 'within' another.

There are four types of nodes in this graph:

- Source node: This is a conceptual node where flow is produced. There is only
  one source node in this graph, so all flow in the graph flows from this one
  node.

- Job nodes: These nodes represent jobs in the schedule. There is an edge with a
  capacity of 1 between the source node and each job node. This edge effectively
  allows each job to be scheduled by using its 1 unit of flow to connect to
  timeslots.

- Satellite timeslot nodes: These nodes represent a timeslot in a satellite. An
  edge between a job and a satellite timeslot node means that the job can be
  completed in the timeslot by the satellite. A single job can have many edges
  to different satellite timeslots, but only has 1 unit of flow flowing into it,
  so it can only have one edge to a satellite timeslot in the optimized graph.

- Sink node: This is a conceptual node where flow is consumed. There is only one
  sink node in this graph, so all flow in the graph flows into this node.

With the exception of the source and sink nodes, the amount of flow entering a
node must equal the amount of flow leaving that node. A path from source to sink
represents a job that is scheduled in a satellite timeslot. The set of all these
paths in the optimized flow network is the schedule of jobs into satellites.

The network flow optimization algorithm happens in two main steps: first the
graph is constructed with nodes and edges with capacities (representing the flow
each edge can carry). Second, a maximum flow optimization algorithm is applied,
which 'fills' edges with flow to maximize the amount of flow flowing through the
graph. There are many such algorithms is provided by the NetworkX Python
library. Once the maximum flow operation is applied, the edges in the graph can
be extracted and formatted into a schedule.
'''


from dataclasses import dataclass
import logging
import logging.config
from pathlib import Path
import time
from typing import List, Optional, Set, Tuple

import networkx as nx
from networkx.exception import NetworkXError
from skyfield.api import EarthSatellite

from soso.debug import debug
from soso.interval_tree import SatelliteInterval
from soso.job import Job
from soso.network_flow.edge_types import \
    Edges, \
    JobToSatelliteTimeSlotEdge, \
    SatelliteToList, \
    SatelliteTimeSlot, \
    SourceToJobEdge, SatelliteTimeSlotToSinkEdge
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


@dataclass
class NetworkFlowResult:
    '''
    A DTO containing the results of the network flow scheduling algorithm.
    '''

    job_to_sat_edges: SatelliteToList[JobToSatelliteTimeSlotEdge]
    '''
    A dictionary mapping each satellite to a list of job-to-timeslot edges.
    '''

    optimized_out_jobs: List[Job]
    '''
    Jobs that could have been scheduled, but were not as part of the
    optimization process.
    '''

    @staticmethod
    def empty(satellites: List[EarthSatellite]) -> 'NetworkFlowResult':
        '''
        Returns an empty `NetworkFlowResult` object with the given satellites.
        This is used for the edge case where there are no jobs to schedule.

        Args:
            satellites: The list of satellites.

        Returns:
            An empty `NetworkFlowResult` object.
        '''
        return NetworkFlowResult({sat: [] for sat in satellites}, [])


def get_scheduled_and_unscheduled_jobs(
    jobs: List[Job],
    jobs_to_sat_edges: SatelliteToList[JobToSatelliteTimeSlotEdge]
) -> Tuple[Set[Job], Set[Job]]:
    '''
    Gets scheduled and unscheduled jobs given edges in a network flow graph.

    A job is considered scheduled if it has an edge to a satellite timeslot.
    Otherwise, it is not scheduled.

    Note that this function can also be used to find schedulable and
    unschedulable jobs (if that's what the underlying network flow graph
    represents).

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


def get_optimized_out_jobs(
    satellite_intervals: SatelliteToList[SatelliteInterval],
    job_to_sat_edges: SatelliteToList[JobToSatelliteTimeSlotEdge]
) -> List[Job]:
    '''
    Gets all jobs that were not scheduled through the optimization process by
    calculating the set difference between jobs in `satellite_intervals` and
    jobs in `job_to_sat_edges`.

    Args:
        satellite_intervals: Dictionary mapping each satellite to intervals from
            the interval tree module.

        job_to_sat_edges: Dictionary mapping each satellite to job-to-timeslot
            edges from this (network flow) module.

    Returns:
        The jobs that were unscheduled due to the optimization process.
    '''

    initial_jobs: Set[Job] = set()
    optimized_jobs: Set[Job] = set()

    # Collect all jobs from satellite intervals
    for intervals in satellite_intervals.values():
        for interval in intervals:
            for job in interval.data:
                initial_jobs.add(job)

    # Collect all scheduled jobs from job-to-timeslot edges
    for edges in job_to_sat_edges.values():
        for edge in edges:
            optimized_jobs.add(edge.job)

    # Calculate the set difference to find unscheduled jobs
    unscheduled_jobs = initial_jobs.difference(optimized_jobs)

    return list(unscheduled_jobs)


@debug
def debug_network_info(
    G: nx.DiGraph,
    unoptimized_edges: Edges,
    optimal_edges: Edges,
    satellite_intervals: SatelliteToList[SatelliteInterval],
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

        satellites: The list of satellites.

        jobs: The list of jobs,

        debug_mode: The debug mode. See the `run_network_flow` function.
    '''

    # Unpack unoptimized edges
    source_to_jobs_edges = unoptimized_edges.sourceToJobEdges
    jobs_to_sat_edges = unoptimized_edges.jobToSatelliteTimeSlotEdges
    sat_to_sink_edges = unoptimized_edges.satelliteTimeSlotToSinkEdges

    schedulable_jobs, unschedulable_jobs = \
        get_scheduled_and_unscheduled_jobs(jobs, jobs_to_sat_edges)

    logger.debug(f'{len(unschedulable_jobs)} not schedulable')

    for job in unschedulable_jobs:
        logger.debug(f'{job} not schedulable')

    # Unpack optimal edges
    new_source_edges = optimal_edges.sourceToJobEdges
    new_jobs_to_sat_edges = optimal_edges.jobToSatelliteTimeSlotEdges
    new_sat_to_sink_edges = optimal_edges.satelliteTimeSlotToSinkEdges

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
        jobs,
        satellites,
        source_to_jobs_edges,
        jobs_to_sat_edges,
        sat_to_sink_edges,
        'All Possible Scheduling Opportunities',
        debug_mode
    )

    # Plot optimal schedule as network flow graph
    plot(
        G,
        satellite_intervals,
        jobs,
        satellites,
        new_source_edges,
        new_jobs_to_sat_edges,
        new_sat_to_sink_edges,
        'Maximum Flow Optimal Schedule',
        debug_mode
    )


def generate_initial_graph_edges(
    satellite_intervals: SatelliteToList[SatelliteInterval],
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
    sat_edges: SatelliteToList[JobToSatelliteTimeSlotEdge] = {
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

    # Edges from satellite time slots to the sink node
    satellite_timeslots_to_sink: SatelliteToList[SatelliteTimeSlotToSinkEdge] = {
        sat: [] for sat in satellites
    }
    for sat, intervals in satellite_intervals.items():
        for interval in intervals:
            satellite_timeslots_to_sink[sat].append(
                SatelliteTimeSlotToSinkEdge(
                    SatelliteTimeSlot(sat, interval.begin, interval.end),
                    'sink',
                    1
                )
            )

    return Edges(
        source_edges,
        sat_edges,
        satellite_timeslots_to_sink
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
    new_sat_edges: SatelliteToList[JobToSatelliteTimeSlotEdge] = {
        sat: [] for sat in satellites
    }
    new_sat_to_sink_edges: SatelliteToList[SatelliteTimeSlotToSinkEdge] = {
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
                    new_sat_to_sink_edges[u.satellite].append(
                        SatelliteTimeSlotToSinkEdge(u, v, flow)
                    )

    return Edges(
        new_source_edges,
        new_sat_edges,
        new_sat_to_sink_edges
    )


def run_network_flow(
    satellites: List[EarthSatellite],
    jobs: List[Job],
    satellite_intervals: SatelliteToList[SatelliteInterval],
    debug_mode: Optional[Path | bool] = None
) -> NetworkFlowResult:
    '''
    The main entry point to the network flow part of the SOSO scheduling
    algorithm.

    Args:
        satellites: The list of satellites to be scheduled with jobs and outage
            requests.

        jobs: The list of jobs to be scheduled.

        satellite_intervals: The dictionary mapping each satellite to its list
            of schedulable intervals.

        debug_mode: Information about how debugging images should be
            handled. If it is set to a `Path` object, images will be outputted
            in that directory. If it is `True`, images will be displayed during
            the execution. Otherwise, if it is `None`, no debugging images will
            be shown or saved at all.

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
        jobs,
        satellites
    )

    # Unpack edges
    source_to_jobs_edges = edges.sourceToJobEdges
    jobs_to_sat_edges = edges.jobToSatelliteTimeSlotEdges
    sat_to_sink_edges = edges.satelliteTimeSlotToSinkEdges

    # Convert job-to-timeslot edges from a dictionary to a list
    job_to_timeslot_edges = [
        edge for edges in jobs_to_sat_edges.values() for edge in edges
    ]

    # Convert satellite-timeslot-to-sink edges from a dictionary to a list
    sink_edges = [
        edge for edges in sat_to_sink_edges.values() for edge in edges
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
    except NetworkXError as e:
        # Sometimes we get the exception that the sink node is not in the graph.
        # In this scenario, it means that there are no edges from the any of the
        # satellite time slots to the sink, so no schedule is possible. We
        # handle that error gracefully here.
        logger.error(f'networkx exception: returning early')
        logger.error(f'networkx exception: {e}')
        return NetworkFlowResult.empty(satellites)

    graph_t1 = time.time()
    logger.debug(f'Finding the maximum flow took {graph_t1 - graph_t0} seconds')

    # Get the optimal edges from the graph
    optimal_edges = extract_optimal_edges(flow_dict, satellites)

    # Get the jobs that were "optimized out"
    optimized_out_jobs = get_optimized_out_jobs(
        satellite_intervals,
        optimal_edges.jobToSatelliteTimeSlotEdges
    )

    alg_t1 = time.time()

    logger.info(
        f'Total network flow runtime (without plotting): {alg_t1 - alg_t0}'
    )

    debug_network_info(
        G,
        edges,
        optimal_edges,
        satellite_intervals,
        satellites,
        jobs,
        debug_mode
    )

    return NetworkFlowResult(
        optimal_edges.jobToSatelliteTimeSlotEdges,
        optimized_out_jobs
    )
