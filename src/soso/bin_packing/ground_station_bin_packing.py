'''
Main functionality for the bin packing algorithm solution to the problem of
scheduling satellite downlinks.

The motivation for using a bin packing algorithm (instead of including this in
the network flow algorithm, for example) is that scheduling downlinks is not a
simple yes/no decision. Different ground station passes can have different
downlink rates and durations, leading to different amounts of data that can be
sent during the downlink. Additionally, different jobs can have different image
sizes, which leads to different amounts of data that satellites have to
downlink.

This problem can be modelled as a bin packing problem. Satellite jobs are
'items' which have a size (the image size in MB), and ground station passes are
'bins' which have a capacity (the amount of data that can be downlinked in the
ground station pass in MB). The objective is to maximize the number of items
(jobs) that are packed into (downlinked in) bins (ground station passes).

The bin packing problem can be solved using linear programming. Google OR-Tools
is used in this algorithm with the SCIP solver to perform the linear programming
optimization. For more information, see
https://developers.google.com/optimization/pack/bin_packing.
'''


from dataclasses import dataclass
import logging
from typing import List, Set

from ortools.linear_solver import pywraplp
from skyfield.api import EarthSatellite

from soso.debug import debug
from soso.interval_tree import GroundStationPassInterval
from soso.job import Job
from soso.network_flow.edge_types import \
    JobToSatelliteTimeSlotEdge, \
    SatelliteTimeSlot, \
    SatelliteToList


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class DownlinkingOpportunities:
    '''
    A data transfer object representing all downlinking opportunities for a
    satellite timeslot.
    '''

    satellite_timeslot: JobToSatelliteTimeSlotEdge
    '''
    The satellite timeslot that is being downlinked.

    This should be returned from the network flow algorithm. It is the list of
    edges from job nodes to satellite timeslot nodes.
    '''

    ground_station_passes: List[GroundStationPassInterval]
    '''
    The list of ground station passes that are able to downlink the job in the
    satellite timeslot.
    '''


@dataclass
class ScheduleUnit:
    '''
    A 'unit' of the schedule. It consists of a job, a timeslot in the satellite
    to complete the job, and a ground station pass to downlink the job.
    '''

    job: Job
    '''
    The job to be completed.
    '''

    job_timeslot: SatelliteTimeSlot
    '''
    The satellite timeslot in which the job will be completed.
    '''

    downlink_timeslot: GroundStationPassInterval
    '''
    The ground station pass where the job will be downlinked.
    '''


@dataclass
class BinPackingResult:
    '''
    The result of the bin packing algorithm.
    '''

    result: SatelliteToList[ScheduleUnit]
    '''
    The solution to the bin packing problem.
    '''

    infeasible_jobs: List[Job]
    '''
    Jobs that could not be scheduled due to infeasibility.
    '''

    @staticmethod
    def empty(satellites: List[EarthSatellite]) -> 'BinPackingResult':
        '''
        Returns an empty `BinPackingResult` object with the given satellites.
        This is used for the edge case where there are no satellite intervals or
        ground station passes to optimize.

        Args:
            satellites: The list of satellites.

        Returns:
            An empty `BinPackingResult` object.
        '''
        return BinPackingResult({sat: [] for sat in satellites}, [])


@debug
def debug_size_and_capacity_information(
    satellite_timeslots: List[JobToSatelliteTimeSlotEdge],
    ground_station_passes: List[GroundStationPassInterval],
    satellite: EarthSatellite
) -> None:
    '''
    Debugs information about the number of jobs, total size of the jobs (in MB),
    and total size of the ground station passes (in MB).
    '''

    # Get set of jobs and total size of all jobs
    jobs_set = set([sat_timeslot.job for sat_timeslot in satellite_timeslots])
    bytes_to_downlink = sum(
        [sat_timeslot.job.size for sat_timeslot in satellite_timeslots]
    )

    # Get total capacity of all ground station passes
    total_capacity_bytes = sum(
        [
            ground_station_pass.capacity
                for ground_station_pass in ground_station_passes
        ]
    )

    logger.debug(
        f'Satellite {satellite.name}: '
            f'{int(bytes_to_downlink / 1_000_000):,} MB to downlink in '
            f'{len(jobs_set)} jobs, '
            f'{int(total_capacity_bytes / 1_000_000):,} MB available in '
            f'{len(ground_station_passes)} ground station passes, '
    )


def get_unpacked_jobs(
    satellite_timeslots: SatelliteToList[JobToSatelliteTimeSlotEdge],
    solution: SatelliteToList[ScheduleUnit]
) -> List[Job]:
    '''
    Gets all jobs that were not scheduled through the optimization process by
    calculating the set difference between jobs in `satellite_timeslots` and
    jobs in `solution` (the solution to the bin packing algorithm).

    Args:
        satellite_timeslots: Dictionary mapping each satellite to
            job-to-timeslot edges from the network flow module.

        solution: The solution to the bin packing problem, which is a full
            schedule of jobs in satellites with downlinking information.

    Returns:
        The list of jobs that were scheduled in the network flow solution, but
        not in the bin packing solution.
    '''

    initial_jobs: Set[Job] = set()

    # Collect all scheduled jobs from job-to-timeslot edges
    for edges in satellite_timeslots.values():
        for edge in edges:
            initial_jobs.add(edge.job)

    bin_packed_jobs: Set[Job] = set()

    # Collect all scheduled jobs from the bin packing solution
    for schedule_units in solution.values():
        for schedule_unit in schedule_units:
            bin_packed_jobs.add(schedule_unit.job)

    return list(initial_jobs.difference(bin_packed_jobs))


def get_downlinkable_jobs(
    downlinking_opportunities: SatelliteToList[DownlinkingOpportunities]
) -> Set[Job]:
    '''
    Gets the set of all downlinkable jobs given a list of downlinking
    opportunities.
    '''

    return set([
        opportunity.satellite_timeslot.job
            for sat, opportunities in downlinking_opportunities.items()
                for opportunity in opportunities
                    if len(opportunities) > 0
    ])


@debug
def debug_downlinkable_jobs(
    satellite_timeslots: SatelliteToList[JobToSatelliteTimeSlotEdge],
    downlinking_opportunities: SatelliteToList[DownlinkingOpportunities]
) -> None:
    '''
    Debugs information about jobs that are not downlinkable.
    '''

    # Get set of all jobs (will be used for logging)
    jobs_set = set(
        [
            timeslot.job
                for sat, timeslots in satellite_timeslots.items()
                    for timeslot in timeslots
        ]
    )

    # Get set of all jobs that have downlinking opportunities
    downlinkable_jobs_set = get_downlinkable_jobs(downlinking_opportunities)

    logger.debug(
        f'Jobs to downlink: {len(jobs_set)}, jobs with downlinking '
            f'opportunities: f{len(downlinkable_jobs_set)}'
    )

    # Get set of all jobs that do not have downlinking opportunities
    non_downlinkable_jobs_set = downlinkable_jobs_set.difference(jobs_set)

    if len(non_downlinkable_jobs_set) > 0:
        logger.debug(
            f'Jobs that do not have downlinking opportunities: '
                f'{non_downlinkable_jobs_set}'
        )



def debug_downlinked_jobs(
    downlinking_opportunities: SatelliteToList[DownlinkingOpportunities],
    solution: SatelliteToList[ScheduleUnit]
) -> None:
    '''
    Debugs information about jobs that were not assigned downlinking times.
    '''

    # Get set of all jobs that have downlinking opportunities
    downlinkable_jobs_set = get_downlinkable_jobs(downlinking_opportunities)

    # Get set of all jobs that were assigned downlinking times
    downlinked_jobs_set = set(
        [
            schedule_unit.job
                for sat, schedule_units in solution.items()
                    for schedule_unit in schedule_units
        ]
    )

    logger.debug(
        f'Jobs scheduled for downlinking: {len(downlinked_jobs_set)} out of '
            f'{len(downlinkable_jobs_set)} downlinkable jobs'
    )

    # Get set of all jobs that were not assigned downlinking times
    non_downlinked_jobs_set = \
        downlinkable_jobs_set.difference(downlinked_jobs_set)

    if len(non_downlinked_jobs_set) > 0:
        logger.debug(
            f'Jobs that were not scheduled downlinks: '
                f'{non_downlinked_jobs_set}'
        )


def schedule_downlinks_for_satellite(
    satellite: EarthSatellite,
    downlinking_opportunities: List[DownlinkingOpportunities]
) -> List[ScheduleUnit]:
    '''
    Attempts to assign a ground station pass for downlinking for each satellite
    timeslot for a satellite.

    Args:
        satellite: The satellite whose downlinks are being scheduled.

        downlinking_opportunities: The downlinking opportunities for a
            satellite.

    Returns:
        A full schedule for a satellite, including jobs, satellite timeslots to
        complete the jobs, and ground station passes to downlink the jobs.
    '''

    # Convert downlinking opportunities to a list of satellite timeslots
    satellite_timeslots = [
        downlinking_opportunity.satellite_timeslot
            for downlinking_opportunity in downlinking_opportunities
    ]

    # Convert downlinking opportunities to a list of ground station passes
    ground_station_passes = list(set(
        ground_station_pass
            for downlinking_opportunity in downlinking_opportunities
                for ground_station_pass in
                    downlinking_opportunity.ground_station_passes
    ))

    debug_size_and_capacity_information(
        satellite_timeslots,
        ground_station_passes,
        satellite
    )

    n = len(satellite_timeslots)
    m = len(ground_station_passes)

    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Decision variables: x, and y

    # x[i, j] = 1, if satellite timeslot i is scheduled to be downlinked in
    #              ground pass timeslot j
    #
    # x[i, j] = 0, otherwise
    x = {}
    for i in range(n):
        for j in range(m):
            x[i, j] = solver.BoolVar(f'x[{i},{j}]')

    # y[j] = 1, if ground station pass timeslot j is used
    #
    # y[j] = 0, otherwise
    y = [solver.BoolVar(f'y[{j}]') for j in range(m)]

    # Constraints

    # Each satellite pass timeslot cannot be assigned to more than one ground
    # station pass timeslot.
    #
    # This is translated into a mathematical constraint as "for each satellite
    # timeslot i, it is scheduled in at most 1 ground station pass".
    for i in range(n):
        solver.Add(sum(x[i, j] for j in range(m)) <= 1)

    # Each ground station pass timeslot has a maximum amount of data that can be
    # downlinked in the pass. The constraint is that the sum of the image data
    # in the satellite timeslots scheduled in each ground station pass cannot
    # exceed the ground station pass's capacity.
    #
    # This is translated into a mathematical constraint as "for each ground
    # station pass j, the sum of the job sizes in the satellite timeslots being
    # downlinked in j should not exceed the capacity of j".
    for j in range(m):
        solver.Add(
            sum(satellite_timeslots[i].job.size * x[i, j] for i in range(n))
                <= ground_station_passes[j].capacity * y[j]
        )

    # Each satellite timeslot can only be downlinked in the ground station
    # passes that it was originally connected to. (This can happen for a variety
    # of reasons, like if the satellite timeslot is chronologically after the
    # ground station pass time.)
    #
    # This information should be given to this function, as this function will
    # not determine whether or not a satellite timeslot can be downlinked in a
    # given ground station pass. This function assumes that each downlinking
    # opportunity in the input has a satellite timeslot and a list of ground
    # station passes that it can be downlinked in.
    #
    # This is translated into a mathematical constraint as "for each satellite
    # timeslot i and ground station pass j, if i cannot be downlinked in j, then
    # x[i, j] = 0".
    for i in range(n):
        for j in range(m):
            downlinking_passes = \
                downlinking_opportunities[i].ground_station_passes
            if ground_station_passes[j] not in downlinking_passes:
                solver.Add(x[i, j] == 0)

    # Objective
    #
    # The objective is to maximize the number of satellite timeslots scheduled
    # into ground station passes.
    #
    # This is represented mathematically as maximizing the sum of x[i, j] for
    # every combination of satellite timeslot i and ground station pass j.
    solver.Maximize(sum(x[i, j] for i in range(n) for j in range(m)))

    # Solve the bin packing problem
    status = solver.Solve()

    solution: List[ScheduleUnit] = []
    unpacked_items = []

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        if status == pywraplp.Solver.OPTIMAL:
            logger.debug('Optimal solution')
        elif status == pywraplp.Solver.FEASIBLE:
            logger.debug('Feasible solution')

        for i in range(n):
            packed = False
            for j in range(m):
                if x[i, j].solution_value() > 0:
                    solution.append(ScheduleUnit(
                        satellite_timeslots[i].job,
                        satellite_timeslots[i].satellite_timeslot,
                        ground_station_passes[j]
                    ))
                    packed = True

            if not packed:
                unpacked_items.append(satellite_timeslots[i])

        logger.debug(f'Unpacked jobs: {len(unpacked_items)}')
        return solution

    else:
        logger.error("No optimal or feasible solution found")
        raise Exception(
            'No optimal of feasible solution found for downlink scheduling'
        )


def schedule_downlinks(
    satellites: List[EarthSatellite],
    satellite_timeslots: SatelliteToList[JobToSatelliteTimeSlotEdge],
    ground_station_passes: SatelliteToList[GroundStationPassInterval]
) -> BinPackingResult:
    '''
    Attempts to assign a ground station pass for downlinking for each satellite
    timeslot.

    Args:
        satellites: The list of satellites.

        satellite_timeslots: The list of satellite timeslots.

        ground_station_passes: The list of ground station passes.

    Returns:
        A full schedule, including jobs, satellite timeslots to complete the
        jobs, and ground station passes to downlink the jobs.
    '''

    if len(satellite_timeslots) == 0 or len(ground_station_passes) == 0:
        logger.info(
            'No satellite timeslots or ground station passes to optimize, '
                'exiting early'
        )
        return BinPackingResult.empty(satellites)

    downlinking_opportunities: SatelliteToList[DownlinkingOpportunities] = {
        sat: [] for sat in satellites
    }

    # Identify all downlinking opportunities for each satellite
    for sat, timeslots in satellite_timeslots.items():
        for timeslot in timeslots:

            downlinks: List[GroundStationPassInterval] = []

            for ground_station_pass in ground_station_passes[sat]:
                # There is a downlinking opportunity between a job and a ground
                # station pass if the job's delivery time is before the
                # beginning of the ground station pass
                if timeslot.job.delivery < ground_station_pass.begin:
                    downlinks.append(ground_station_pass)

            downlinking_opportunities[sat].append(
                DownlinkingOpportunities(
                    timeslot,
                    downlinks
                )
            )

    solution: SatelliteToList[ScheduleUnit] = {sat: [] for sat in satellites}

    for satellite, opportunities in downlinking_opportunities.items():
        # Schedule downlinks
        satellite_schedule = schedule_downlinks_for_satellite(
            satellite,
            opportunities
        )

        solution[satellite] = satellite_schedule

    # Get jobs that were "not packed" (a.k.a. could have been scheduled but were
    # not as part of the optimization algorithm)
    unpacked_jobs = get_unpacked_jobs(satellite_timeslots, solution)

    return BinPackingResult(
        solution,
        unpacked_jobs
    )
