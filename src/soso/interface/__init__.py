'''
Definitions of the interface to the scheduling algorithm. Since the algorithm
has many different parts (like network flow, bin packing, genetic algorithm),
this package defines a simplified interface for the user.
'''


from typing import cast, Dict, List

from skyfield.api import EarthSatellite, load

from soso.genetic.genetic_scheduler import \
    optimize_schedule, \
    run_genetic_algorithm
from soso.interval_tree import generate_satellite_intervals
from soso.persistence import \
    get_persister, \
    ScheduleNotFoundException
from soso.interface.interface_exceptions import \
    ExistingScheduleNotFoundException, \
    MissingInputHashException
from soso.interface.interface_types import \
    PlannedOrder, \
    ScheduleOutput, \
    ScheduleParameters
from soso.interface.interface_utils import consolidate_inputs, get_ephemeris


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
        params.ground_station_outage_requests,
        ts,
        eph
    )

    good_jobs = [job for intervals in satellite_passes.satellite_intervals.values() for interval in intervals for job in interval.data]

    perfect_genetic_result = optimize_schedule(
        satellites,
        good_jobs,
        satellite_passes.satellite_intervals,
        satellite_passes.ground_station_passes
    )

    # Run the genetic algorithm to generate the schedule
    genetic_algorithm_result = run_genetic_algorithm(
        satellites,
        good_jobs,
        # params.jobs,
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

    planned_jobs = set(order.job for orders in planned_orders.values() for order in orders)

    rejected_orders = list(
        set(good_jobs)
            .difference(satellite_passes.unschedulable_jobs)
            .difference(satellite_passes.unschedulable_after_outages_jobs)
            .difference(satellite_passes.unschedulable_after_ground_station_jobs)
            .difference(perfect_genetic_result.undownlinkable_jobs)
            .difference(planned_jobs)
    )

    schedule_output = ScheduleOutput(
        input_hash=ScheduleOutput.convert_to_hash(params),
        impossible_orders=
            satellite_passes.unschedulable_jobs,
        impossible_orders_from_outages=
            satellite_passes.unschedulable_after_outages_jobs,
        impossible_orders_from_ground_stations=
            satellite_passes.unschedulable_after_ground_station_jobs,
        undownlinkable_orders=
            perfect_genetic_result.undownlinkable_jobs,
        rejected_orders=
            rejected_orders,
        planned_orders=
            planned_orders
    )

    persister = get_persister()
    persister.persist(params, schedule_output)

    return schedule_output


def rerun_full(params: ScheduleParameters) -> ScheduleOutput:
    '''
    Re-runs the scheduling algorithm.

    Args:
        params: The input parameters of the scheduling algorithm, including the
            hash of the previous scheduler run's input parameters.

    Returns:
        The output of the scheduling algorithm whose input is the combination of
        the previous scheduling run's input parameters and the new input
        parameters.
    '''

    if params.input_hash is None:
        raise MissingInputHashException()

    persister = get_persister()

    try:
        existing_schedule = persister.retrieve(params.input_hash)
    except ScheduleNotFoundException as e:
        raise ExistingScheduleNotFoundException()

    new_params = consolidate_inputs(existing_schedule, params)

    return run(new_params)
