'''
Main functionality for the network flow optimization solution to the
job-satellite scheduling problem.
'''


from datetime import timezone
import json
import logging
import logging.config
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from pathlib import Path
import time
from typing import cast, Dict, List, Set
from intervaltree import IntervalTree
from skyfield.api import EarthSatellite, load, Time, wgs84
from soso.debug import debug
from soso.job import Job
from soso.network_flow.edge_types import \
    Edges, \
    JobToSatelliteTimeSlotEdge, \
    SatelliteTimeSlotToSinkEdge, \
    SourceToJobEdge
from soso.network_flow.plot import plot
from soso.outage_request import OutageRequest
from soso.utils import counter_generator


eph = load('de421.bsp')

ts = load.timescale()

altitude_degrees = 30.0

event_names = f'rise above {altitude_degrees}°', \
    'culminate', \
    f'set below {altitude_degrees}°'

logging.config.fileConfig('logging_config.ini')
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


def parse_jobs(order_data_dir: Path) -> List[Job]:
    '''
    Parses orders from a directory of JSON files.

    Args:
        order_data_dir: The file path of the directory containing the JSON files
            representing the orders.

    Returns:
        The list of parsed orders.
    '''
    jobs = []
    counter = counter_generator()

    for filename in os.listdir(order_data_dir):
        if filename.endswith('.json'):
            full_path = order_data_dir / filename
            with open(full_path, 'r') as f:
                data = json.load(f)
                if next(counter) > 15:
                    continue
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


def parse_outage_requests(
        outage_request_data_dir: Path,
        satellites: List[EarthSatellite]
    ) -> List[OutageRequest]:
    '''
    Parses outage requests from a directory of JSON files.

    Args:
        outage_request_data_dir: The file path of the directory containing the
            JSON files representing the outage requests.

        satellites: The list of satellites. This will be used to add the
            satellite reference to the outage request for that satellite.

    Returns:
        The list of parsed outage requests.
    '''
    outage_requests: List[OutageRequest] = []
    counter = counter_generator()
    sat_name_to_sat = {sat.name: sat for sat in satellites}

    for filename in os.listdir(outage_request_data_dir):
        if filename.endswith('.json'):
            full_path = outage_request_data_dir / filename
            with open(full_path, 'r') as f:
                data = json.load(f)
                outage_request = OutageRequest(
                    f'Outage Request {next(counter)}',
                    sat_name_to_sat[data['Satellite']],
                    data['OutageStartTime'],
                    data['OutageEndTime']
                )
                outage_requests.append(outage_request)

    return outage_requests


def parse_satellites(satellite_data_dir: Path) -> List[EarthSatellite]:
    '''
    Parses satellites from a directory of JSON files.

    Args:
        satellite_data_dir: The file path of the directory containing the JSON
            files representing the satellites.

    Returns:
        The list of parsed satellites.
    '''
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


def update_trees_with_jobs(
        trees: Dict[EarthSatellite, IntervalTree],
        sat: EarthSatellite,
        job: Job,
        t0: Time,
        t1: Time
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
    '''
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


def generate_trees(
        satellites: List[EarthSatellite],
        jobs: List[Job],
        outage_requests: List[OutageRequest],
        t0: Time,
        t1: Time
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
            update_trees_with_jobs(trees, sat, job, t0, t1)

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
        trees: Dict[EarthSatellite, IntervalTree],
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
        trees: The `dict` mapping each satellite to its interval tree.

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
    source_edges = [SourceToJobEdge('source', job.name, 1) for job in jobs]

    # Edges from jobs to timeslots.
    #
    # Remember: for each interval in the interval tree, we held the start and
    # end time of the interval, along with all the jobs that could be scheduled
    # in that interval.
    #
    # This for loop just creates edges between each job and the intervals that
    # it could be scheduled in.
    job_to_timeslot_edges = []
    sat_edges: Dict[EarthSatellite, List[JobToSatelliteTimeSlotEdge]] = {
        sat: [] for sat in satellites
    }
    for sat, tree in trees.items():
        for interval in tree:
            for job in interval.data:
                sat_edges[sat].append(
                    JobToSatelliteTimeSlotEdge(
                        job.name,
                        f'{sat.name} {interval.begin} {interval.end}',
                        1
                    )
                )
                job_to_timeslot_edges.append(
                    (job.name, f'{sat.name} {interval.begin} {interval.end}', 1)
                )

    # Edges from timeslots to the sink
    sat_to_sink_edges: Dict[EarthSatellite, List[SatelliteTimeSlotToSinkEdge]] = {
        sat: [] for sat in satellites
    }
    for sat, tree in trees.items():
        for interval in tree:
            sat_to_sink_edges[sat].append(
                SatelliteTimeSlotToSinkEdge(
                    f'{sat.name} {interval.begin} {interval.end}',
                    'sink',
                    1
                )
            )

    return Edges(source_edges, sat_edges, sat_to_sink_edges)


def extract_optimal_edges(
        flow_dict: dict,
        jobs: List[Job],
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
    new_sat_edges: Dict[EarthSatellite, List[JobToSatelliteTimeSlotEdge]] = {sat: [] for sat in satellites}
    new_sat_to_sink_edges: Dict[EarthSatellite, List[SatelliteTimeSlotToSinkEdge]] = {sat: [] for sat in satellites}

    for u in flow_dict:
        for v, flow in flow_dict[u].items():
            if isinstance(u, str) and u == 'source':
                if flow > 0:
                    new_source_edges.append(SourceToJobEdge(u,v,1))
            elif isinstance(u, str) and u in [job.name for job in jobs]:
                if flow > 0:
                    if v.startswith('SOSO-1'):
                        new_sat_edges[satellites[0]].append(JobToSatelliteTimeSlotEdge(u,v,1))
                    elif v.startswith('SOSO-2'):
                        new_sat_edges[satellites[1]].append(JobToSatelliteTimeSlotEdge(u,v,1))
                    elif v.startswith('SOSO-3'):
                        new_sat_edges[satellites[2]].append(JobToSatelliteTimeSlotEdge(u,v,1))
                    elif v.startswith('SOSO-4'):
                        new_sat_edges[satellites[3]].append(JobToSatelliteTimeSlotEdge(u,v,1))
                    elif v.startswith('SOSO-5'):
                        new_sat_edges[satellites[4]].append(JobToSatelliteTimeSlotEdge(u,v,1))
            elif isinstance(u, str) and u.startswith('SOSO'):
                if flow > 0:
                    if u.startswith('SOSO-1'):
                        new_sat_to_sink_edges[satellites[0]].append(SatelliteTimeSlotToSinkEdge(u,v,1))
                    elif u.startswith('SOSO-2'):
                        new_sat_to_sink_edges[satellites[1]].append(SatelliteTimeSlotToSinkEdge(u,v,1))
                    elif u.startswith('SOSO-3'):
                        new_sat_to_sink_edges[satellites[2]].append(SatelliteTimeSlotToSinkEdge(u,v,1))
                    elif u.startswith('SOSO-4'):
                        new_sat_to_sink_edges[satellites[3]].append(SatelliteTimeSlotToSinkEdge(u,v,1))
                    elif u.startswith('SOSO-5'):
                        new_sat_to_sink_edges[satellites[4]].append(SatelliteTimeSlotToSinkEdge(u,v,1))

    return Edges(new_source_edges, new_sat_edges, new_sat_to_sink_edges)


def main():
    parse_t0 = time.time()

    # === GET SATELLITES AND ORDERS DATA ===
    # Define directories containing the satellites and orders data
    project_dir = Path(os.path.dirname(__file__)).parent.parent.parent
    data_dir = project_dir / 'data'
    order_data_dir = data_dir / 'orders'
    satellite_data_dir = data_dir / 'satellites'
    outage_request_data_dir = data_dir / 'outages'

    # Parse satellites and orders data
    satellites = parse_satellites(satellite_data_dir)
    jobs = parse_jobs(order_data_dir)
    outage_requests = parse_outage_requests(outage_request_data_dir, satellites)

    parse_t1 = time.time()
    logger.info(f'Parsing data took {parse_t1 - parse_t0} seconds')

    tree_t0 = time.time()

    # === DEFINE START AND END TIMES ===
    # The scheduling will cover the start of the earliest job to the end of the
    # latest job
    t0 = ts.from_datetime(min(job.start for job in jobs).replace(tzinfo=timezone.utc))
    t1 = ts.from_datetime(max(job.end for job in jobs).replace(tzinfo=timezone.utc))

    # === GENERATE INTERVAL TREES ===
    # One interval tree per satellite, where each interval in the tree holds the
    # jobs that could be scheduled in that interval
    trees = generate_trees(satellites, jobs, outage_requests, t0, t1)

    tree_t1 = time.time()
    logger.info(f'Making the interval tree took {tree_t1 - tree_t0} seconds')

    graph_t0 = time.time()

    # === MODELING PROBLEM AS GRAPH AND GETTING MAXIMUM FLOW ===
    # Create an empty directed graph
    G = nx.DiGraph()

    # Generate edges for the graph
    # source_to_jobs_edges, jobs_to_sat_edges, sat_to_sink_edges = \
    edges = \
        generate_initial_graph_edges(trees, jobs, satellites)
    source_to_jobs_edges = edges.sourceToJobEdges
    jobs_to_sat_edges = edges.jobToSatelliteTimeSlotEdges
    sat_to_sink_edges = edges.satelliteTimeSlotToSinkEdges

    sink_edges = [sat_to_sink_edge for sat_edges in sat_to_sink_edges.values() for sat_to_sink_edge in sat_edges]
    job_to_timeslot_edges = [job_to_timeslot_edge for sat_edges in jobs_to_sat_edges.values() for job_to_timeslot_edge in sat_edges]

    graph_edges = source_to_jobs_edges + job_to_timeslot_edges + sink_edges

    # Add edges with capacities to the graph
    for u, v, capacity in graph_edges:
        G.add_edge(u, v, capacity=capacity)

    # Define source and sink nodes
    source = 'source'
    sink = 'sink'

    # Calculate the maximum flow and the flow on each edge
    flow_value, flow_dict = nx.maximum_flow(G, source, sink)

    graph_t1 = time.time()
    logger.info(f'Finding the maximum flow took {graph_t1 - graph_t0} seconds')

    # Get the optimal edges from the graph
    # new_source_edges, new_sat_edges, new_sat_to_sink_edges = \
    optimal_edges = \
        extract_optimal_edges(flow_dict, jobs, satellites)
    new_source_edges = optimal_edges.sourceToJobEdges
    new_sat_edges = optimal_edges.jobToSatelliteTimeSlotEdges
    new_sat_to_sink_edges = optimal_edges.satelliteTimeSlotToSinkEdges

    new_job_to_timeslot_edges = []
    for sat_edges in jobs_to_sat_edges.values():
        for job_to_timeslot_edge in sat_edges:
            new_job_to_timeslot_edges.append(job_to_timeslot_edge)

    all_jobs = set([job.name for job in jobs])
    scheduled_jobs = set()

    for job_timeslot in new_job_to_timeslot_edges:
        scheduled_jobs.add(job_timeslot[0])
        logger.info(f'{job_timeslot[0]} at {job_timeslot[1]}')

    unscheduled_jobs = all_jobs.difference(scheduled_jobs)

    for unscheduled_job in unscheduled_jobs:
        logger.info(f'{unscheduled_job} not scheduled')

    logger.info(f'Out of {len(all_jobs)} jobs, {len(scheduled_jobs)} were scheduled, {len(unscheduled_jobs)} were not')
    logger.info(f'Total runtime (without plotting): {sum([parse_t1-parse_t0, tree_t1-tree_t0, graph_t1-graph_t0])}')

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


if __name__ == '__main__':
    main()
