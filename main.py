'''
Main script for running the scheduler from the command line in different ways.
'''


import argparse
import json
import logging
import logging.config
logging.config.fileConfig('logging_config.ini')
import os
from pathlib import Path
import requests
import time

from skyfield.api import load

from soso.genetic.genetic_scheduler import run_genetic_algorithm
from soso.bin_packing.ground_station_bin_packing import schedule_downlinks
from soso.network_flow.network_flow_scheduler import run_network_flow
from soso.utils import \
    parse_ground_stations, \
    parse_jobs, \
    parse_outage_requests, \
    parse_satellites, \
    parse_two_line_elements, \
    print_api_result, \
    print_bin_packing_result, \
    print_genetic_result, \
    print_network_flow_result, \
    SchedulerServer
from soso.interface import ScheduleOutput, ScheduleParameters
from soso.interval_tree import generate_satellite_intervals


logger: logging.Logger = logging.getLogger(__name__)

# Load timescale and ephemeris
ts = load.timescale()
eph = load('de421.bsp')

parse_t0 = time.time()

# Define directories containing the satellites and orders data
project_dir = Path(os.path.dirname(__file__))
data_dir = project_dir / 'data'
order_data_dir = data_dir / 'orders'
satellite_data_dir = data_dir / 'satellites'
outage_request_data_dir = data_dir / 'outages'
ground_station_data_dir = data_dir / 'ground_stations'

# Parse satellites and orders data
two_line_elements = parse_two_line_elements(satellite_data_dir)
satellites = parse_satellites(satellite_data_dir, ts)
jobs = parse_jobs(order_data_dir)
outage_requests = parse_outage_requests(outage_request_data_dir, satellites)
ground_stations = parse_ground_stations(ground_station_data_dir)

parse_t1 = time.time()
logger.info(f'Parsing data took {parse_t1 - parse_t0} seconds')

# Command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('alg_type', type=str)
args = parser.parse_args()

if args.alg_type in ['network', 'bin', 'genetic']:

    # Generate intervals for each satellite
    satellite_passes = generate_satellite_intervals(
        satellites,
        jobs,
        outage_requests,
        ground_stations,
        ts,
        eph
    )
    satellite_intervals = satellite_passes.satellite_intervals
    ground_station_passes = satellite_passes.ground_station_passes

    # Network flow algorithm
    if args.alg_type == 'network':
        alg_t0 = time.time()

        solution = run_network_flow(satellites, jobs, satellite_intervals, True)

        alg_t1 = time.time()

        print_network_flow_result(solution)

        jobs_scheduled = sum(
            [len(timeslots) for timeslots in solution.job_to_sat_edges.values()]
        )
        print(
            f'{jobs_scheduled} out of {len(jobs)} jobs scheduled in '
                f'{alg_t1 - alg_t0} seconds'
        )

    # Network flow and bin packing algorithm
    elif args.alg_type == 'bin':
        alg_t0 = time.time()

        network_flow_solution = run_network_flow(
            satellites,
            jobs,
            satellite_intervals,
            True
        )
        solution = schedule_downlinks(
            satellites,
            network_flow_solution.job_to_sat_edges,
            ground_station_passes
        )

        alg_t1 = time.time()

        print_bin_packing_result(solution)

        jobs_scheduled = sum(
            [len(schedule_units) for schedule_units in solution.result.values()]
        )
        print(
            f'{jobs_scheduled} out of {len(jobs)} jobs scheduled in '
                f'{alg_t1 - alg_t0} seconds'
        )

    # Genetic algorithm (implicitly uses both network flow and bin packing
    # algorithms)
    elif args.alg_type == 'genetic':
        alg_t0 = time.time()

        solution = run_genetic_algorithm(
            satellites,
            jobs,
            outage_requests,
            satellite_intervals,
            ground_station_passes
        )

        alg_t1 = time.time()

        print_genetic_result(solution)

        jobs_scheduled = sum(
            [len(schedule_units) for schedule_units in solution.result.values()]
        )
        print(
            f'{jobs_scheduled} out of {len(jobs)} jobs scheduled in '
                f'{alg_t1 - alg_t0} seconds'
        )

elif args.alg_type == 'api':
    try:
        server = SchedulerServer()

        server.start()

        params = ScheduleParameters(
            two_line_elements=two_line_elements,
            jobs=jobs,
            ground_stations=ground_stations,
            outage_requests=outage_requests
        )

        alg_t0 = time.time()

        # Send a test request
        response = requests.post(
            "http://127.0.0.1:8000/schedule",
            headers={"Content-Type": "application/json"},
            json=json.loads(params.model_dump_json())
        )

        if response.status_code == 200:
            alg_t1 = time.time()

            time.sleep(1)

            solution = ScheduleOutput.model_validate(response.json())

            print_api_result(solution)


            jobs_scheduled = sum([
                len(planned_orders)
                    for planned_orders in solution.planned_orders.values()
            ])
            print(
                f'{jobs_scheduled} out of {len(jobs)} jobs scheduled in '
                    f'{alg_t1 - alg_t0} seconds'
            )
        else:
            print(f'Response status: {response.status_code}')
            print(response.content)

    except Exception as e:
        print('Error in API call')
        print(e)

    finally:
        server.stop()

else:
    raise Exception('Invalid command line argument option')
