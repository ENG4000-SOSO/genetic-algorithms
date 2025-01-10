'''
Main functionality for the network flow optimization solution to the
job-satellite scheduling problem.
'''


from datetime import timezone
import logging
import logging.config
import time
from typing import cast, Dict, List, Set

from intervaltree import Interval, IntervalTree
import networkx as nx
from skyfield.api import EarthSatellite, Loader, Time, Timescale, wgs84

from soso.debug import debug
from soso.job import Job
from soso.network_flow.edge_types import \
    Edges, \
    JobToSatelliteTimeSlotEdge, \
    SatelliteTimeSlot, \
    SatelliteTimeSlotToSinkEdge, \
    SourceToJobEdge
from soso.network_flow.plot import plot
from soso.outage_request import OutageRequest


ALTITUDE_DEGREES = 30.0
'''
Threshold for the degrees above the horizontal for a satellite to be considered
to have a point on Earth in its field-of-view.
'''

logger: logging.Logger = logging.getLogger(__name__)


def get_unschedulable_events(
        trees: Dict[EarthSatellite, IntervalTree],
        jobs: List[Job]
    ) -> Set[Job]:
    '''
    Gets all jobs that are not schedulable.

    A job is not schedulable if it is not in any of the satellites' interval
    trees.
    '''
    schedulable_job_set: Set[Job] = set()
    all_job_set: Set[Job] = set(jobs)
    for tree in trees.values():
        for interval in tree:
            for job in cast(Set[Job], interval.data):
                schedulable_job_set.add(job)
    unscheduled_job_set = all_job_set.difference(schedulable_job_set)
    return unscheduled_job_set


@debug
def log_interval_trees(
        trees: Dict[EarthSatellite, IntervalTree],
        jobs: List[Job]
    ):
    '''
    Logs jobs that are not schedulable.
    '''
    logger = logging.getLogger(__name__)
    logger.debug(
        'Time slots found before outages: %d',
        sum(len(tree) for tree in trees.values())
    )
    unschedulable_job_set = get_unschedulable_events(trees, jobs)
    if len(unschedulable_job_set) == 0:
        logger.debug(
            'All jobs can be scheduled in timeslots in the interval trees'
        )
    else:
        logger.debug(
            'The following jobs could not be added to the interval tree: %s',
            unschedulable_job_set
        )


@debug
def log_interval_trees_after_outages(
        trees: Dict[EarthSatellite, IntervalTree],
        jobs: List[Job]
    ):
    '''
    Logs jobs that are not schedulable after considering outage requests.
    '''
    logger = logging.getLogger(__name__)
    logger.debug(
        'Time slots found after outages: %d',
        sum(len(tree) for tree in trees.values())
    )
    unscheduled_job_set = get_unschedulable_events(trees, jobs)
    if len(unscheduled_job_set) == 0:
        logger.debug(
            'All jobs can still be scheduled even after considering outage '
                'requests'
        )
    else:
        logger.debug(
            'The following jobs were removed from the interval tree due to '
                'outage requests: %s',
            unscheduled_job_set
        )


def update_trees_with_jobs(
        trees: Dict[EarthSatellite, IntervalTree],
        sat: EarthSatellite,
        job: Job,
        t0: Time,
        t1: Time,
        eph: Loader
    ) -> None:
    '''
    Updates interval tree of then given satellite, if possible, with the given
    job.

    If the given satellite passes over the job's area on Earth between the start
    and end times, then it is added to the satellite's interval tree.

    Args:
        trees: The `dict` mapping satellites to their interval tree.

        sat: The satellite being checked for schedulability with the job.

        job: The job being checked for schedulability with the satellite.

        t0: The start time of the window that is being checked.

        t1: The end time of the window that is being checked.

        eph: The ephemeris data used to perform astronomical calculations.
    '''
    # Location to be imaged
    location = wgs84.latlon(job.latitude, job.longitude)

    # Find all times that the location can be imaged by the satellite
    t, events = sat.find_events(
        location,
        t0,
        t1,
        altitude_degrees=ALTITUDE_DEGREES
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


def generate_trees(
        satellites: List[EarthSatellite],
        jobs: List[Job],
        outage_requests: List[OutageRequest],
        t0: Time,
        t1: Time,
        eph: Loader
    ) -> Dict[EarthSatellite, IntervalTree]:
    '''
    Generates interval trees for each satellite representing schedulable jobs.

    The interval trees generated for each satellite contain intervals with a
    start and end time of the interval and the jobs that can potentially be
    scheduled in that interval.

    Args:
        satellites: The full list of satellites.

        jobs: The full list of jobs.

        outage_requests: The full list of outage requests.

        t0: The start time of the space mission.

        t0: The end time of the space mission.

        eph: The ephemeris data used to perform astronomical calculations.

    Returns:
        The `dict` mapping satellites to their interval tree, where the interval
        trees contain intervals that represent a window of time and the jobs
        that can potentially be scheduled for that satellite in that time.
    '''

    # This is an interval tree.
    #
    # Each node of the tree stores an interval, which has:
    #   - start time
    #   - end time
    #   - data (containing the job that is able to be scheduled in this
    #     interval)
    trees = {sat: IntervalTree() for sat in satellites}

    # Update every satellite's interval tree with the jobs it can have scheduled
    for sat in satellites:
        for job in jobs:
            update_trees_with_jobs(trees, sat, job, t0, t1, eph)

    # Merge overlaps in each interval tree.
    #
    # Whenever two intervals overlap, we create a new interval for them that
    # accommodates them both, and the jobs that can be scheduled in that
    # interval is the union of the jobs that could have been scheduled in the
    # original intervals.
    for sat, tree in trees.items():
        tree.merge_overlaps(data_reducer=lambda x, y: x.union(y))

    log_interval_trees(trees, jobs)

    # Remove intervals that overlap with outage requests
    for outage_request in outage_requests:
        trees[outage_request.satellite].remove_overlap(
            outage_request.start,
            outage_request.end
        )

    log_interval_trees_after_outages(trees, jobs)

    return trees


def generate_initial_graph_edges(
        satellite_intervals: Dict[EarthSatellite, List[Interval]],
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
        A tuple containing

        1. A list of edges from the source to jobs, represented as tuples of 
            `(source, job, flow)`, where `flow` indicates the amount of flow.

        2. A dictionary mapping each satellite to a list of edges from jobs to
            that  satellite time slot, represented as tuples of
            `(job, satellite time slot, flow)`, where `flow`  indicates the
            amount of flow.

        3. A dictionary mapping each satellite to a list of edges from that
            satellite  to the sink, represented as tuples of
            `(satellite time slot, sink, flow)`, where `flow` indicates the
            amount of flow.
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

        jobs: The complete list of jobs.

        satellites: The complete list of satellites.

    Returns:
        A tuple containing
        1. A list of edges from the source to jobs, represented as tuples of 
            `(source, job, flow)`, where `flow` indicates the amount of flow.
        2. A dictionary mapping each satellite to a list of edges from jobs to
            that  satellite time slot, represented as tuples of
            `(job, satellite time slot, flow)`, where `flow`  indicates the
            amount of flow.
        3. A dictionary mapping each satellite to a list of edges from that
            satellite  to the sink, represented as tuples of
            `(satellite time slot, sink, flow)`, where `flow` indicates the
            amount of flow.

        Note that the flow will always be 1 because edges with 0 flow are
        ignored and the maximum capacity of every edge is 1.
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


def format_solution(
        satellites: List[EarthSatellite],
        satellite_job_edge_dict: Dict[EarthSatellite, List[JobToSatelliteTimeSlotEdge]]
    ) -> Dict[EarthSatellite, List[str]]:
    '''
    Formats the solution in an understandable way for other functionality to use
    more easily.

    Args:
        satellites: List of satellites (only used to initialize the formatted
        solution).

        satellite_job_edge_dict: Dictionary mapping each satellite to the
        optimal edge lists for it, where each edge connects a job to one of the
        satellite's time slots.

    Returns:
        Dictionary mapping each satellite to a list of string representations of
        a job and a time slot for the job.
    '''

    solution: Dict[EarthSatellite, List[str]] = { sat: [] for sat in satellites }

    for satellite, job_to_satellite_time_slots in satellite_job_edge_dict.items():
        # print(f'Satellite: {satellite.name}')
        for job_to_satellite_time_slot in job_to_satellite_time_slots:
            solution[satellite].append(f'{job_to_satellite_time_slot.job} {job_to_satellite_time_slot.satellite_timeslot}')
            # print(f'    {job_to_satellite_time_slot.job} @ {job_to_satellite_time_slot.satellite_timeslot.split()}')

    return solution


def generate_satellite_intervals(
    satellites: List[EarthSatellite],
    jobs: List[Job],
    outage_requests: List[OutageRequest],
    ts: Timescale,
    eph: Loader
) -> Dict[EarthSatellite, List[Interval]]:
    '''
    Generates intervals for each satellite where each interval contains a set of
    jobs that can be chosen to be scheduled between the interval's start and end
    times.

    Args:
        satellites: The list of satellites to be scheduled with jobs and outage
        requests.

        jobs: The list of jobs to be scheduled.

        outage_requests: The list of (non-negotiable) outage requests.

        ts: The Skyfield timescale being used to simulate events in the future.

        eph: The Skyfield ephemeris data being used to perform astronomical
        calculations.

    Returns:
        A dictionary mapping each satellite to a list, where items in the list
        are intervals. Each interval has a `begin` and an `end` time properties,
        and a `data` property that is a set of jobs that can be scheduled in that
        interval (only one job in the set will be chosen to be scheduled in the
        interval).
    '''

    # Empty jobs edge case
    if len(jobs) == 0:
        logger.info('Empty jobs list, returning early')
        return {sat: [] for sat in satellites}

    tree_t0 = time.time()

    # Define start and end times.
    # The scheduling will cover the start of the earliest job to the end of the
    # latest job.
    t0 = ts.from_datetime(
        min(job.start for job in jobs).replace(tzinfo=timezone.utc)
    )
    t1 = ts.from_datetime(
        max(job.end for job in jobs).replace(tzinfo=timezone.utc)
    )

    # Generate interval trees.
    # One interval tree per satellite, where each interval in the tree holds the
    # jobs that could be scheduled in that interval.
    trees = generate_trees(satellites, jobs, outage_requests, t0, t1, eph)

    tree_t1 = time.time()
    logger.debug(f'Making the interval trees took {tree_t1 - tree_t0} seconds')

    # Convert the interval trees to lists of intervals to simplify the interface
    return {sat: cast(List[Interval], list(tree)) for sat, tree in trees.items()}


def run_network_flow(
    satellites: List[EarthSatellite],
    jobs: List[Job],
    satellite_intervals: Dict[EarthSatellite, List[Interval]]
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

    # Format and return solution
    # return format_solution(satellites, new_sat_edges)
    return new_sat_edges

# def run_network_flow(
#     satellites: List[EarthSatellite],
#     jobs: List[Job],
#     outage_requests: List[OutageRequest],
#     ts: Timescale,
#     eph: Loader
# ) -> Dict[EarthSatellite, List[JobToSatelliteTimeSlotEdge]]:
#     '''
#     The main entry point to the network flow part of the SOSO scheduling
#     algorithm.

#     Args:
#         satellites: The list of satellites to be scheduled with jobs and outage
#         requests.

#         jobs: The list of jobs to be scheduled.

#         outage_requests: The list of (non-negotiable) outage requests.

#         ts: The Skyfield timescale being used to simulate events in the future.

#         eph: The Skyfield ephemeris data being used to perform astronomical
#         calculations.

#     Returns:
#         A dictionary mapping each satellite to a list, where items in the list
#         are representations of a job scheduled in a time slot.
#     '''
#     pass
