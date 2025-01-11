'''
Interval tree functionality for the SOSO algorithm.

Interval trees are a crucial part of the scheduling algorithm. They are used to
efficiently generate timeslots for each satellite and handle overlaps of
timeslots.
'''


from datetime import timezone
import logging
import logging.config
import time
from typing import cast, Dict, List, Set

from intervaltree import IntervalTree
from skyfield.api import EarthSatellite, Loader, Time, Timescale, wgs84

from soso.debug import debug
from soso.interval_tree import SatelliteInterval
from soso.job import Job
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


def generate_satellite_intervals(
    satellites: List[EarthSatellite],
    jobs: List[Job],
    outage_requests: List[OutageRequest],
    ts: Timescale,
    eph: Loader
) -> Dict[EarthSatellite, List[SatelliteInterval]]:
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
    satellite_intervals: Dict[EarthSatellite, List[SatelliteInterval]] = {
        sat: [] for sat in satellites
    }
    for sat, tree in trees.items():
        for interval in tree:
            satellite_intervals[sat].append(
                SatelliteInterval(
                    interval.begin,
                    interval.end,
                    list(interval.data)
                )
            )
    return satellite_intervals
