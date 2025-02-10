'''
Utilities to be used throughout the SOSO project.
'''


import json
import os
from pathlib import Path
from typing import List

from skyfield.api import EarthSatellite, Timescale

from soso.job import GroundStation
from soso.job import Job
from soso.outage_request import OutageRequest


def counter_generator():
    '''
    Generates a counter to make counting easier and more Pythonic.

    To initialize the counter:

    ```
    counter = counter_generator()
    ```

    then, increment the counter:

    ```
    next(counter)
    ```
    '''
    i = 0
    while True:
        yield i
        i += 1


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

                # if next(counter) > 15:
                #     continue

                job = Job(
                    f'Job {next(counter)}',
                    data['ImageStartTime'],
                    data['ImageEndTime'],
                    data['DeliveryTime'],
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


def parse_satellites(
    satellite_data_dir: Path,
    ts: Timescale
) -> List[EarthSatellite]:
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


def parse_ground_stations(ground_station_data_dir: Path) -> List[GroundStation]:
    '''
    Parses ground stations from a directory of JSON files.

    Args:
        ground_station_data_dir: The file path of the directory containing the
        JSON files representing the ground stations.

    Returns:
        The list of parsed ground stations.
    '''

    ground_stations: List[GroundStation] = []

    for filename in sorted(os.listdir(ground_station_data_dir)):
        if filename.endswith('.json'):
            full_path = ground_station_data_dir / filename
            with open(full_path, 'r') as f:
                data = json.load(f)
                ground_station = GroundStation(
                    data['name'],
                    data['latitude'],
                    data['longitude'],
                    data['height'],
                    data['mask'],
                    data['uplink_rate'],
                    data['downlink_rate']
                )
                ground_stations.append(ground_station)

    return ground_stations
