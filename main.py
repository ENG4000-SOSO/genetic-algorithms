import logging
import logging.config
logging.config.fileConfig('logging_config.ini')
import os
from pathlib import Path
import time

from skyfield.api import load

import soso.genetic.genetic_scheduler
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

# solution = run_network_flow(satellites, jobs, satellite_intervals)
solution = soso.genetic.genetic_scheduler.run_genetic_algorithm(satellites, jobs, outage_requests, satellite_intervals)

for satellite, job_to_satellite_time_slots in solution.items():
    print(f'Satellite: {satellite.name}')
    for job_to_satellite_time_slot in job_to_satellite_time_slots:
        print(f'    {job_to_satellite_time_slot.job} @ {job_to_satellite_time_slot.satellite_timeslot.start} - {job_to_satellite_time_slot.satellite_timeslot.end}')
