'''
Definitions of the interface to the scheduling algorithm. Since the algorithm
has many different parts (like network flow, bin packing, genetic algorithm),
this package defines a simplified interface for the user.
'''


from datetime import datetime
from pathlib import Path
from typing import cast, Dict, List

from pydantic import BaseModel
from skyfield.api import EarthSatellite, load

from soso.bin_packing.ground_station_bin_packing import ScheduleUnit
from soso.genetic.genetic_scheduler import run_genetic_algorithm
from soso.interval_tree import generate_satellite_intervals
from soso.job import Job, GroundStation, TwoLineElement
from soso.outage_request import OutageRequest


def get_ephemeris():
    '''
    Gets the `de421.bsp` ephemeris from the current working directory. If the
    ephemeris does not exist in the current working directory, or if it is not a
    file, then an exception is thrown.
    '''

    eph_file = Path.cwd() / 'de421.bsp'

    if eph_file.exists():
        if eph_file.is_file():
            return load(str(eph_file.absolute()))
        else:
            raise Exception(
                f'{eph_file.name} found in directory {eph_file.parent} but is '
                    'not a file'
            )
    else:
        raise Exception(
            f'{eph_file.name} not found in directory {eph_file.parent}'
        )


class ScheduleParameters(BaseModel):
    '''
    Representation of the parameters of the scheduling algorithm.
    '''

    two_line_elements: List[TwoLineElement]
    '''
    The list of satellites to be scheduled with orders.
    '''

    jobs: List[Job]
    '''
    The orders to be scheduled into satellites.
    '''

    ground_stations: List[GroundStation]
    '''
    The ground stations than can downlink orders from satellites.
    '''

    outage_requests: List[OutageRequest]
    '''
    The outage requests that add constraints to when satellites are unavailable.
    '''


class PlannedOrder(BaseModel):
    '''
    Representation of a planned order.
    '''

    job: Job
    '''
    The job being planned in the order.
    '''

    satellite_name: str
    '''
    The satellite that is being planned to fulfill the order.
    '''

    ground_station_name: str
    '''
    The ground station that is being planned to downlink the order.
    '''

    job_begin: datetime
    '''
    The start time of the interval in which the satellite will complete the job.
    '''

    job_end: datetime
    '''
    The end time of the interval in which the satellite will complete the job.
    '''

    downlink_begin: datetime
    '''
    The start time of the interval in which the job will be downlinked.
    '''

    downlink_end: datetime
    '''
    The end time of the interval in which the job will be downlinked.
    '''


class ScheduleOutput(BaseModel):
    '''
    A representation of the result of the entire scheduling algorithm.
    '''

    impossible_orders: List[Job]
    '''
    Jobs that were not scheduled because they were just not possible to be
    fulfilled.
    '''

    impossible_orders_from_outages: List[Job]
    '''
    Jobs that were not scheduled because the only times they could have been
    scheduled were blocked by outages.
    '''

    impossible_orders_from_ground_stations: List[Job]
    '''
    Jobs that were not scheduled because there was a lack of availability of
    ground stations to downlink them.
    '''

    rejected_orders: List[Job]
    '''
    Jobs that could have been scheduled but were not as part of the optimization
    algorithm.
    '''

    planned_orders: Dict[str, List[PlannedOrder]]
    '''
    Jobs that have been successfully scheduled.
    '''


def run(params: ScheduleParameters) -> ScheduleOutput:
    '''
    Runs the full scheduling algorithm.

    Args:
        params: The input parameters of the scheduling algorithm, including the
            jobs, satellites, ground stations, and outage requests.

    Returns:
        The output of the scheduling algorithm, including the planned orders,
            rejected orders, and orders that are impossible to fulfill.
    '''

    ts = load.timescale()
    eph = get_ephemeris()

    # Convert two-line-elements to satellite objects
    satellites = [
        EarthSatellite(tle.line1, tle.line2, tle.name)
            for tle in params.two_line_elements
    ]

    # Generate a dictionary mapping satellite names to satellite objects
    sat_name_to_sat: Dict[str, EarthSatellite] = {
        cast(str, sat.name): sat for sat in satellites
    }

    # Assign satellite objects to outage requests
    for outage_request in params.outage_requests:
        outage_request.assign_satellite(sat_name_to_sat)

    # Generate schedulable time intervals for each satellite and ground station
    satellite_passes = generate_satellite_intervals(
        satellites,
        params.jobs,
        params.outage_requests,
        params.ground_stations,
        ts,
        eph
    )

    # Run the genetic algorithm to generate the schedule
    genetic_algorithm_result = run_genetic_algorithm(
        satellites,
        params.jobs,
        params.outage_requests,
        satellite_passes.satellite_intervals,
        satellite_passes.ground_station_passes
    )

    # Convert the genetic algorithm result to the output format (which can be
    # easily serialized/deserialized by Pydantic)
    planned_orders: Dict[str, List[PlannedOrder]] = {
        cast(str, satellite.name): [
            PlannedOrder(
                job=
                    schedule_unit.job,
                satellite_name=
                    cast(str, schedule_unit.job_timeslot.satellite.name),
                ground_station_name=
                    schedule_unit.downlink_timeslot.ground_station.name,
                job_begin=
                    schedule_unit.job_timeslot.start,
                job_end=
                    schedule_unit.job_timeslot.end,
                downlink_begin=
                    schedule_unit.downlink_timeslot.begin,
                downlink_end=
                    schedule_unit.downlink_timeslot.end
            )
            for schedule_unit in schedule_units
        ]
        for satellite, schedule_units in genetic_algorithm_result.result.items()
    }

    return ScheduleOutput(
        impossible_orders=
            satellite_passes.unschedulable_jobs,
        impossible_orders_from_outages=
            satellite_passes.unschedulable_after_outages_jobs,
        impossible_orders_from_ground_stations=
            satellite_passes.unschedulable_after_ground_station_jobs,
        rejected_orders=
            genetic_algorithm_result.optimized_out_jobs,
        planned_orders=
            planned_orders
    )
