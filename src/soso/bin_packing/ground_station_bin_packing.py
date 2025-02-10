'''
Bin packing algorithm.
'''


from dataclasses import dataclass
import logging
from typing import List

from ortools.linear_solver import pywraplp
from skyfield.api import EarthSatellite

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


def schedule_downlinks_for_satellite(
    satellite: EarthSatellite,
    downlinking_opportunities: List[DownlinkingOpportunities]
) -> List[ScheduleUnit]:
    '''
    Attempts to assign a ground station pass for downlinking for each satellite
    timeslot for a satellite.

    Args:
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
                for ground_station_pass in downlinking_opportunity.ground_station_passes
    ))

    jobs_set = set([a.job for a in satellite_timeslots])
    bytes_to_downlink = sum([a.job.size for a in satellite_timeslots])
    total_capacity = sum([a.capacity for a in ground_station_passes])
    logger.debug(f'Satellite {satellite.name}: {len(jobs_set)} jobs, {int(bytes_to_downlink / 1_000_000):,} MB to downlink, {int(total_capacity / 1_000_000):,} MB available')

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
            print(satellite_timeslots[i].satellite_timeslot)
            print(len(satellite_timeslots))
            print(downlinking_opportunities[i].satellite_timeslot.satellite_timeslot)
            print(len(downlinking_opportunities))
            print(downlinking_opportunities[i].ground_station_passes)
            if satellite_timeslots[i].satellite_timeslot == downlinking_opportunities[i].satellite_timeslot.satellite_timeslot:
                print('good')
            else:
                raise Exception('AS')

            if ground_station_passes[j] not in downlinking_opportunities[i].ground_station_passes:
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

    solution = []
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
        raise Exception('bad!')


def schedule_downlinks(
    satellites: List[EarthSatellite],
    satellite_timeslots: SatelliteToList[JobToSatelliteTimeSlotEdge],
    ground_station_passes: SatelliteToList[GroundStationPassInterval]
) -> SatelliteToList[ScheduleUnit]:
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

    downlinking_opportunities: SatelliteToList[DownlinkingOpportunities] = {
        sat: [] for sat in satellites
    }

    print('***************')
    print([p.ground_station for s, g in ground_station_passes.items() for p in g])
    jobs_before = set([a.job for sat, s in satellite_timeslots.items() for a in s])

    for sat, timeslots in satellite_timeslots.items():
        for timeslot in timeslots:

            downlinks: List[GroundStationPassInterval] = []

            for ground_station_pass in ground_station_passes[sat]:
                if timeslot.job.delivery < ground_station_pass.begin:
                    downlinks.append(ground_station_pass)

            downlinking_opportunities[sat].append(
                DownlinkingOpportunities(
                    timeslot,
                    downlinks
                )
            )

    jobs_after = set([a.satellite_timeslot.job for sat, s in downlinking_opportunities.items() for a in s])

    logger.warning(f'jobs before: {len(jobs_before)}')
    logger.warning(f'jobs after: {len(jobs_after)}')
    logger.warning(f'lost these: {jobs_after.difference(jobs_before)}')
    logger.warning(f'should be empty: {jobs_before.difference(jobs_after)}')

    solution: SatelliteToList[ScheduleUnit] = {sat: [] for sat in satellites}

    for satellite, x in downlinking_opportunities.items():
        solution[satellite] = schedule_downlinks_for_satellite(satellite, x)

    jobs_optimized = set([a.job for sat, s in solution.items() for a in s])
    logger.warning(f'jobs optimized: {len(jobs_optimized)}')
    logger.warning(f'lost to optimization: {jobs_after.difference(jobs_optimized)}')

    return solution
