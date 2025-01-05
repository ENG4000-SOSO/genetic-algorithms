import logging
import os
from pathlib import Path
import time
from skyfield.api import load
from soso.network_flow.network_flow_scheduler_improved import run
from soso.utils import parse_jobs, parse_outage_requests, parse_satellites


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

# Parse satellites and orders data
satellites = parse_satellites(satellite_data_dir, ts)
jobs = parse_jobs(order_data_dir)
outage_requests = parse_outage_requests(outage_request_data_dir, satellites)

parse_t1 = time.time()
logger.info(f'Parsing data took {parse_t1 - parse_t0} seconds')

solution = run(satellites, jobs, outage_requests, ts, eph)

print(solution)
