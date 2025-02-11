'''
Interval tree functionality for the SOSO algorithm.

Interval trees are a crucial part of the scheduling algorithm. They are used to
efficiently generate timeslots for each satellite and handle overlaps of
timeslots.
'''


from dataclasses import dataclass
import datetime
import logging
import logging.config
import time
from typing import Any, cast, Dict, List, Optional, Set, Tuple

from intervaltree import IntervalTree
from skyfield.api import EarthSatellite, Time, Timescale, wgs84

from soso.debug import debug
from soso.interval_tree import GroundStationPassInterval, SatelliteInterval
from soso.job import GroundStation, Job, SatellitePassLocation
from soso.outage_request import OutageRequest


@dataclass
class SatellitePasses:
    '''
    A DTO containing information about satellites and their passes over specific
    locations.
    '''

    satellite_intervals: Dict[EarthSatellite, List[SatelliteInterval]]
    '''
    A dictionary mapping each satellite to a list of intervals where each
    interval contains begin and end times and a list of jobs that can each be
    scheduled in the satellite within the begin and end times.
    '''

    ground_station_passes: Dict[EarthSatellite, List[GroundStationPassInterval]]
    '''
    A dictionary mapping each satellite to a list of intervals where each
    interval contains being and end times and a ground station that the
    satellite passes over within the being and end times.
    '''


ALTITUDE_DEGREES = 30.0
'''
Threshold for the degrees above the horizontal for a satellite to be considered
to have a point on Earth in its field-of-view.
'''

GROUND_STATION_ALTITUDE_DEGREES = 0.0
'''
Threshold for the degrees above the horizontal for a satellite to be considered
able to communicate to a ground station.
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


def find_pass_events(
    sat: EarthSatellite,
    satellite_pass_location: SatellitePassLocation,
    t0: Time,
    t1: Time,
    eph: Any,
    altitude_degrees: float
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    '''
    Finds events of when a satellite passes over a location on Earth.

    Args:
        sat: The satellite being analyzed.

        satellite_pass_location: The location on Earth being checked for
            satellite passes.

        t0: The start time of the space mission.

        t0: The end time of the space mission.

        eph: The ephemeris data used to perform astronomical calculations.

    Returns:
        A list of the pass events.

        Each element of the list is a tuple with a begin and end time,
        representing the beginning and ending, respectively, of the satellite
        pass.
    '''

    # Initialize list of events
    found_events: List[Tuple[datetime.datetime, datetime.datetime]] = []

    # Location to be imaged
    location = wgs84.latlon(
        satellite_pass_location.latitude, satellite_pass_location.longitude
    )

    # Find all times that the location can be imaged by the satellite
    t, events = sat.find_events(
        location,
        t0,
        t1,
        altitude_degrees=altitude_degrees
    )

    # Check which of the imaging times are in sunlight
    sunlit = sat.at(t).is_sunlit(eph)
    start: Optional[datetime.datetime] = None
    end: Optional[datetime.datetime] = None

    for ti, event, sunlit_flag in zip(t, events, sunlit):
        if sunlit_flag:
            if event == 0:
                start = cast(datetime.datetime, ti.utc_datetime())
            elif event == 2:
                if not start:
                    continue
                end = cast(datetime.datetime, ti.utc_datetime())
                found_events.append((start, end))
                start = None
                end = None
        else:
            start = None
            end = None

    return found_events


def generate_ground_station_pass_intervals(
    satellites: List[EarthSatellite],
    ground_stations: List[GroundStation],
    t0: Time,
    t1: Time,
    eph: Any
) -> Dict[EarthSatellite, List[GroundStationPassInterval]]:
    '''
    Generates intervals for each satellite where each interval contains a ground
    station that the satellite passes over between the interval's start and end
    times.

    Args:
        satellites: The list of satellites to be analyzed for ground station
            passes.

        ground_stations: The list of ground stations.

        t0: The start time of the space mission.

        t0: The end time of the space mission.

        eph: The Skyfield ephemeris data being used to perform astronomical
            calculations.

    Returns:
        A dictionary mapping each satellite to a list, where items in the list
        are intervals. Each interval has a `begin` and an `end` time properties,
        and a `ground_station` property that is a location on Earth that the
        satellite will pass over within the interval.
    '''

    # Initialize dictionary of ground station pass intervals
    ground_station_pass_intervals: Dict[EarthSatellite, List[GroundStationPassInterval]] = {
        sat: [] for sat in satellites
    }

    for sat in satellites:
        for ground_station in ground_stations:
            # Find when the satellite passes over the ground station
            events = find_pass_events(
                sat,
                ground_station,
                t0,
                t1,
                eph,
                GROUND_STATION_ALTITUDE_DEGREES
            )

            # Add the pass events to the dictionary
            for begin, end in events:
                ground_station_pass_intervals[sat].append(
                    GroundStationPassInterval(begin, end, ground_station)
                )

    return ground_station_pass_intervals


def update_trees_with_jobs(
    trees: Dict[EarthSatellite, IntervalTree],
    sat: EarthSatellite,
    job: Job,
    t0: Time,
    t1: Time,
    eph: Any
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

    # Find pass events for the satellite and job
    events = find_pass_events(sat, job, t0, t1, eph, ALTITUDE_DEGREES)

    # Add pass events to the interval tree
    for begin, end in events:
        trees[sat].addi(begin, end, set([job]))


def generate_trees(
    satellites: List[EarthSatellite],
    jobs: List[Job],
    outage_requests: List[OutageRequest],
    t0: Time,
    t1: Time,
    eph: Any
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
    #   - data (containing the jobs that are able to be scheduled in this
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


def get_start_and_end_times_of_mission(jobs: List[Job], ts: Timescale):
    '''
    Extracts the start and end times of the entire mission from a list of jobs
    that must be scheduled in the mission.

    Args:
        jobs: The list of jobs to be scheduled.

        ts: The Skyfield timescale being used to simulate events in the future.

    Returns:
        The start and end times of the mission, where the start time of the
        mission is the start time of the earliest job, and the end time of the
        mission is the end time of the earliest job.
    '''

    # Start time is the earliest start time of all the jobs
    t0 = ts.from_datetime(
        min(job.start for job in jobs).replace(tzinfo=datetime.timezone.utc)
    )

    # End time is the latest end time of all the jobs
    t1 = ts.from_datetime(
        max(job.end for job in jobs).replace(tzinfo=datetime.timezone.utc)
            + datetime.timedelta(days=2)
    )

    return t0, t1


def convert_trees_to_satellite_intervals(
    satellites: List[EarthSatellite],
    trees: Dict[EarthSatellite, IntervalTree]
) -> Dict[EarthSatellite, List[SatelliteInterval]]:
    '''
    Converts interval trees to a list of satellite intervals.

    This function exists to convert the output format of the interval tree to a
    more standardized form for other functions to consume easier.

    Args:
        satellites: The list of satellites in the space mission.

        trees: A dictionary mapping each satellite to an interval tree
            representing intervals that jobs can be scheduled in.

    Returns:
        A dictionary mapping each satellite to a list of intervals representing
        intervals that jobs can be scheduled in.
    '''

    # Initialize dictionary of satellite intervals
    satellite_intervals: Dict[EarthSatellite, List[SatelliteInterval]] = {
        sat: [] for sat in satellites
    }

    # Copy each interval in the interval tree to the dictionary of interval
    # lists
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


def filter_unschedulable_timeslots(
    satellite_intervals: Dict[EarthSatellite, List[SatelliteInterval]],
    ground_station_passes: Dict[EarthSatellite, List[GroundStationPassInterval]]
) -> Dict[EarthSatellite, List[SatelliteInterval]]:
    '''
    Filters out satellite intervals that cannot be downlinked in a ground
    station pass.

    Args:
        satellite_intervals: Dictionary mapping each satellite to intervals.

        ground_station_passes: Dictionary mapping each satellite to a ground
            station passes.

    Returns:
        Satellite intervals that can be downlinked by a ground station pass.
    '''

    new_satellite_intervals = {sat: [] for sat in satellite_intervals.keys()}

    for sat, intervals in satellite_intervals.items():
        for interval in intervals:
            schedulable = False
            for ground_station_pass in ground_station_passes[sat]:
                if interval.end < ground_station_pass.begin:
                    schedulable = True
                    break
            if schedulable:
                new_satellite_intervals[sat].append(interval)
            else:
                logger.debug(
                    f'Removing Interval {interval} because there are no ground '
                        'station passes for it'
                )

    return new_satellite_intervals


def generate_satellite_intervals(
    satellites: List[EarthSatellite],
    jobs: List[Job],
    outage_requests: List[OutageRequest],
    ground_stations: List[GroundStation],
    ts: Timescale,
    eph: Any
) -> SatellitePasses:
    '''
    Generates intervals for each satellite where each interval contains a set of
    jobs that can be chosen to be scheduled between the interval's start and end
    times.

    Args:
        satellites: The list of satellites to be scheduled with jobs and outage
            requests.

        jobs: The list of jobs to be scheduled.

        outage_requests: The list of (non-negotiable) outage requests.

        ground_stations: The list of ground stations.

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
        return SatellitePasses(
            {sat: [] for sat in satellites},
            {sat: [] for sat in satellites}
        )

    tree_t0 = time.time()

    # Define start and end times.
    # The scheduling will cover the start of the earliest job to the end of the
    # latest job.
    t0, t1 = get_start_and_end_times_of_mission(jobs, ts)

    # Generate interval trees.
    # One interval tree per satellite, where each interval in the tree holds the
    # jobs that could be scheduled in that interval.
    trees = generate_trees(satellites, jobs, outage_requests, t0, t1, eph)

    # Generate ground station passes.
    ground_station_passes = generate_ground_station_pass_intervals(
        satellites,
        ground_stations,
        t0,
        t1,
        eph
    )

    tree_t1 = time.time()
    logger.debug(f'Making the interval trees took {tree_t1 - tree_t0} seconds')

    # Convert the interval trees to lists of intervals to simplify the interface
    unfiltered_satellite_intervals = \
        convert_trees_to_satellite_intervals(satellites, trees)

    satellite_intervals = filter_unschedulable_timeslots(
        unfiltered_satellite_intervals,
        ground_station_passes
    )

    # Return satellite intervals and ground station passes in a DTO
    return SatellitePasses(satellite_intervals, ground_station_passes)
