import argparse
import logging
import logging.config
logging.config.fileConfig('logging_config.ini')
import os
from pathlib import Path
import time

from skyfield.api import load

from soso.genetic.genetic_scheduler import run_genetic_algorithm
from soso.bin_packing.ground_station_bin_packing import schedule_downlinks
from soso.network_flow.network_flow_scheduler_improved import run_network_flow
from soso.utils import \
    parse_ground_stations, \
    parse_jobs, \
    parse_outage_requests, \
    parse_satellites
from soso.interval_tree import generate_satellite_intervals


logger: logging.Logger = logging.getLogger(__name__)

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
satellites = parse_satellites(satellite_data_dir, ts)
jobs = parse_jobs(order_data_dir)
outage_requests = parse_outage_requests(outage_request_data_dir, satellites)
ground_stations = parse_ground_stations(ground_station_data_dir)

parse_t1 = time.time()
logger.info(f'Parsing data took {parse_t1 - parse_t0} seconds')

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

parser = argparse.ArgumentParser()
parser.add_argument('alg_type', type=str)
args = parser.parse_args()

a = time.time()

if args.alg_type == 'network':
    solution = run_network_flow(satellites, jobs, satellite_intervals, ground_station_passes, True)
    for satellite, timeslots in solution.items():
        print(f'Satellite: {satellite.name}')
        for timeslot in timeslots:
            print(f'    {timeslot.job} in {timeslot.satellite_timeslot.satellite.name} from {timeslot.satellite_timeslot.start.strftime("%B %d %Y @ %I:%M %p")} to {timeslot.satellite_timeslot.end.strftime("%B %d %Y @ %I:%M %p")}, downlinked at {"?"} from {"?"} to {"?"}')

elif args.alg_type == 'bin':
    solution_part1 = run_network_flow(satellites, jobs, satellite_intervals, ground_station_passes, True)
    solution = schedule_downlinks(satellites, solution_part1, ground_station_passes)
    print('=======')
    for satellite, dtos in solution.items():
        print(f'Satellite: {satellite.name}')
        for dto in dtos:
            print(f'    {dto.job} in {dto.job_timeslot.satellite.name} from {dto.job_timeslot.start.strftime("%B %d %Y @ %I:%M %p")} to {dto.job_timeslot.end.strftime("%B %d %Y @ %I:%M %p")}, downlinked at {dto.downlink_timeslot.ground_station.name} from {dto.downlink_timeslot.begin.strftime("%B %d %Y @ %I:%M %p")} to {dto.downlink_timeslot.end.strftime("%B %d %Y @ %I:%M %p")}')

elif args.alg_type == 'genetic':
    solution = run_genetic_algorithm(satellites, jobs, outage_requests, satellite_intervals, ground_station_passes)
    print('=======')
    for satellite, dtos in solution.items():
        print(f'Satellite: {satellite.name}')
        for dto in dtos:
            print(f'    {dto.job} in {dto.job_timeslot.satellite.name} from {dto.job_timeslot.start.strftime("%B %d %Y @ %I:%M %p")} to {dto.job_timeslot.end.strftime("%B %d %Y @ %I:%M %p")}, downlinked at {dto.downlink_timeslot.ground_station.name} from {dto.downlink_timeslot.begin.strftime("%B %d %Y @ %I:%M %p")} to {dto.downlink_timeslot.end.strftime("%B %d %Y @ %I:%M %p")}')

else:
    raise Exception('Invalid command line argument option')

b = time.time()

print(f'{sum([len(timeslotlist) for timeslotlist in solution.values()])} out of {len(jobs)} scheduled in {b-a:.2f} seconds')
print(f'{b-a} seconds')
