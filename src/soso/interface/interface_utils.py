'''
Utility functions to be used for ease of interfacing with the scheduler.
'''


from pathlib import Path

from skyfield.api import load

from soso.persistence import SchedulingInputOutputData
from soso.interface.interface_types import ScheduleParameters


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


def consolidate_inputs(
    existing_schedule: SchedulingInputOutputData,
    new_params: ScheduleParameters
) -> ScheduleParameters:
    '''
    Consolidates a previous scheduler run's input parameters with new
    parameters.

    Args:
        existing_schedule: The previous scheduler run's input parameters and
            output.

        new_params: The new parameters to combine with the previous scheduling
            run.

    Returns:
        Consolidated input parameters, where each attribute of the scheduler
        parameters is the set union of that attribute for the existing and new
        parameters.
    '''

    consolidated_two_line_elements = list(
        set(existing_schedule.params.two_line_elements).union(set(new_params.two_line_elements))
    )
    consolidated_jobs = list(
        set(existing_schedule.params.jobs).union(set(new_params.jobs))
    )
    consolidated_ground_stations = list(
        set(existing_schedule.params.ground_stations).union(set(new_params.ground_stations))
    )
    consolidated_outage_requests = list(
        set(existing_schedule.params.outage_requests).union(set(new_params.outage_requests))
    )
    consolidated_ground_station_outage_requests = list(
        set(existing_schedule.params.ground_station_outage_requests).union(set(new_params.ground_station_outage_requests))
    )

    return ScheduleParameters(
        input_hash=None,
        two_line_elements=consolidated_two_line_elements,
        jobs=consolidated_jobs,
        ground_stations=consolidated_ground_stations,
        outage_requests=consolidated_outage_requests,
        ground_station_outage_requests=consolidated_ground_station_outage_requests
    )
