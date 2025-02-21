'''
Utilities to be used to simplify the running and testing of the SOSO algorithm.
'''


import json
import os
from pathlib import Path
import select
import subprocess
import threading
import time
from typing import cast, Dict, List, Optional

from colorama import Fore, init as colorama_init
from skyfield.api import EarthSatellite, Timescale

from soso.bin_packing.ground_station_bin_packing import \
    BinPackingResult, \
    ScheduleUnit
from soso.genetic.genetic_scheduler import GeneticAlgorithmResult
from soso.interface import ScheduleOutput
from soso.job import GroundStation
from soso.job import Job, TwoLineElement
from soso.network_flow.edge_types import SatelliteToList
from soso.network_flow.network_flow_scheduler import NetworkFlowResult
from soso.outage_request import OutageRequest


FORMAT = '%B %d %Y @ %I:%M %p'
'''
Format string for dates.
'''

TAB = '    '
'''
Tab character to be used when formatting text.
'''

# Make sure colorama resets the color code after every print statement
colorama_init(autoreset=True)


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

                c = next(counter)
                # if c > 15:
                #     continue

                job = Job(
                    name=f'Job {c}',
                    start=data['ImageStartTime'],
                    end=data['ImageEndTime'],
                    delivery=data['DeliveryTime'],
                    priority=data['Priority'],
                    latitude=data['Latitude'],
                    longitude=data['Longitude']
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
    sat_name_to_sat: Dict[str, EarthSatellite] = {
        cast(str, sat.name): sat for sat in satellites
    }

    for filename in os.listdir(outage_request_data_dir):
        if filename.endswith('.json'):
            full_path = outage_request_data_dir / filename
            with open(full_path, 'r') as f:
                data = json.load(f)
                outage_request = OutageRequest(
                    name=f'Outage Request {next(counter)}',
                    satellite_name=data['Satellite'],
                    start=data['OutageStartTime'],
                    end=data['OutageEndTime']
                )
                outage_request.assign_satellite(sat_name_to_sat)
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


def parse_two_line_elements(
    satellite_data_dir: Path
) -> List[TwoLineElement]:
    '''
    Parses satellites from a directory of JSON files.

    Args:
        satellite_data_dir: The file path of the directory containing the JSON
            files representing the satellites.

    Returns:
        The list of parsed satellites.
    '''
    two_line_elements: List[TwoLineElement] = []

    for filename in sorted(os.listdir(satellite_data_dir)):
        if filename.endswith('.json'):
            full_path = satellite_data_dir / filename
            with open(full_path, 'r') as f:
                data = json.load(f)
                tle = TwoLineElement(
                    name=data['name'],
                    line1=data['line1'],
                    line2=data['line2'],
                )
                two_line_elements.append(tle)

    return two_line_elements


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
                    name=data['name'],
                    latitude=data['latitude'],
                    longitude=data['longitude'],
                    height=data['height'],
                    mask=data['mask'],
                    uplink_rate=data['uplink_rate'],
                    downlink_rate=data['downlink_rate']
                )
                ground_stations.append(ground_station)

    return ground_stations


def print_network_flow_result(result: NetworkFlowResult) -> None:
    '''
    Prints the result of the network flow scheduling algorithm.

    Args:
        result: The result of the network flow scheduling algorithm, containing
            the job-to-satellite timeslot edges for each satellite.
    '''

    for satellite, timeslots in result.job_to_sat_edges.items():
        print(f'Satellite: {satellite.name}')
        for timeslot in timeslots:
            job_start = timeslot.satellite_timeslot.start.strftime(FORMAT)
            job_end = timeslot.satellite_timeslot.end.strftime(FORMAT)
            print(
                f'{TAB}{timeslot.job} in '
                    f'{timeslot.satellite_timeslot.satellite.name} '
                    f'from {job_start} to {job_end}, '
                    f'downlinked at {"?"} from {"?"} to {"?"}'
            )


def print_full_schedule(result: SatelliteToList[ScheduleUnit]):
    '''
    Prints the full schedule of jobs for each satellite.

    Args:
        result: A dictionary mapping each satellite to a list of schedule
            units, where each unit contains the job timeslot and the downlink
            timeslot.
    '''

    for satellite, dtos in result.items():
        print(f'Satellite: {satellite.name}')
        for dto in dtos:
            job_start = dto.job_timeslot.start.strftime(FORMAT)
            job_end = dto.job_timeslot.end.strftime(FORMAT)
            downlink_start = dto.downlink_timeslot.begin.strftime(FORMAT)
            downlink_end = dto.downlink_timeslot.end.strftime(FORMAT)
            print(
                f'{TAB}{dto.job} in {dto.job_timeslot.satellite.name} '
                    f'from {job_start} to {job_end}, '
                    f'downlinked at {dto.downlink_timeslot.ground_station.name} '
                    f'from {downlink_start} to {downlink_end}'
            )


def print_bin_packing_result(result: BinPackingResult) -> None:
    '''
    Prints the result of the bin packing scheduling algorithm.

    Args:
        result: The result of the bin packing scheduling algorithm, containing
            the schedule units (job and downlink times) for each satellite.
    '''
    print_full_schedule(result.result)


def print_genetic_result(result: GeneticAlgorithmResult) -> None:
    '''
    Prints the result of the genetic algorithm scheduling.

    Args:
        result: The result of the genetic algorithm scheduling, containing the
            schedule units (job and downlink times) for each satellite.
    '''
    print_full_schedule(result.result)


def print_api_result(result: ScheduleOutput):
    '''
    Prints the result of the API response of the full scheduling algorithm.

    Args:
        result: The result of the full scheduling algorithm, containing the
            planned orders (job and downlink times) for each satellite.
    '''

    for satellite_name, dtos in result.planned_orders.items():
        print(f'Satellite: {satellite_name}')
        for dto in dtos:
            job_start = dto.job_begin.strftime(FORMAT)
            job_end = dto.job_end.strftime(FORMAT)
            downlink_start = dto.downlink_begin.strftime(FORMAT)
            downlink_end = dto.downlink_end.strftime(FORMAT)
            print(
                f'{TAB}{dto.job} in {dto.satellite_name} '
                    f'from {job_start} to {job_end}, '
                    f'downlinked at {dto.ground_station_name} '
                    f'from {downlink_start} to {downlink_end}'
            )
    print(f'Impossible orders: {result.impossible_orders}')
    print(
        f'Impossible orders from ground stations: '
            f'{result.impossible_orders_from_ground_stations}'
    )
    print(
        f'Impossible orders from outages: '
            f'{result.impossible_orders_from_outages}'
    )
    print(f'Rejected orders: {result.rejected_orders}')


class SchedulerServer:
    '''
    Manages the lifecycle of the scheduling algorithm FastAPI server, including
    starting, stopping, and printing output from the server.

    The server will run as a subprocess in another thread to allow the main
    thread to stay unblocked as API calls take place.
    '''

    stop_flag: bool
    '''
    Stop flag signals to the thread running the server to stop the server.
    '''

    thread: Optional[threading.Thread]
    '''
    The thread that the server will be run in.
    '''

    process: Optional[subprocess.Popen]
    '''
    The subprocess that will run the server.
    '''

    def __init__(self):
        self.stop_flag = False
        self.process = None
        self.thread = None

    def _run(self):
        '''
        Runs the scheduling FastAPI server in a subprocess, printing both stdout
        and stderr from the subprocess.
        '''

        print(Fore.CYAN + 'stdout from the server will be printed in cyan')
        print(
            Fore.LIGHTRED_EX + 'stderr from the server will be printed in red'
        )

        # Start the FastAPI server
        self.process = subprocess.Popen(
            ['uvicorn', 'soso.api:app', '--log-level', 'debug'],
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if self.process.stdout is None:
            raise Exception('No standard output')
        if self.process.stderr is None:
            raise Exception('No standard error')

        # Boolean the represents whether or not the previous iteration of the
        # while loop captured output from the server
        no_output = False

        while True:
            # Check if the server produced output
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr],
                [],
                [],
                0.5
            )

            no_output = True

            # Print the output from the server, if there is output to print
            for stream in rlist:
                output = stream.readline().decode().strip()
                if output:
                    # There was output, so set the no_output flag to false
                    no_output = False
                    if stream is self.process.stdout:
                        print(Fore.CYAN + str(output))
                    elif stream is self.process.stderr:
                        print(Fore.LIGHTRED_EX + str(output))

            # Break out of the loop if the server process has terminated and the
            # last iteration of the while loop produced no output (this ensures
            # we don't miss any output from the server as it's shutting down)
            if self.process.poll() is not None and no_output:
                break

            if self.stop_flag:
                # Check if the server is still running
                if self.process.poll() is None:
                    # Stop the server
                    self.process.terminate()
                    self.process.wait()

                    # Wait for the server to stop
                    time.sleep(1)

        # Process has terminated, so remove it
        self.process = None

    def start(self):
        '''
        Starts the scheduling FastAPI server in another thread.
        '''

        if self.thread is not None and self.thread.is_alive():
            print('Server is already running')
            return

        # Run the server in another thread
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        # Give the server time to start
        time.sleep(5)

    def stop(self):
        '''
        Stops the scheduling FastAPI server process and the thread it is running
        in.
        '''

        if self.thread is None:
            print('Server is not running')
            return

        # Stop the server
        self.stop_flag = True
        self.thread.join()
        self.thread = None
