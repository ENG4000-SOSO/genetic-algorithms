'''
Driver for running the "day in the life" dataset as input to the scheduler.
'''


import datetime
import logging
import logging.config
logging.config.fileConfig('logging_config.ini')
import os
from pathlib import Path
from typing import cast, Dict

from colorama import Fore, init as colorama_init
from skyfield.api import EarthSatellite, load

from soso.utils import \
    parse_ground_stations, \
    parse_jobs, \
    parse_satellites, \
    parse_two_line_elements, \
    parse_maintenance_orders, \
    print_api_result
from soso.interface import rerun_full, run, ScheduleParameters
from soso.outage_request import GroundStationOutageRequest, OutageRequest


# Make sure colorama resets the color code after every print statement
colorama_init(autoreset=True)

logger: logging.Logger = logging.getLogger(__name__)

# Load timescale and ephemeris
ts = load.timescale()
eph = load('de421.bsp')

project_dir = Path(os.path.dirname(__file__))
day_in_the_life_data_dir = project_dir / 'day_in_the_life_data'
data_dir = project_dir / 'data'
order_data_dir = data_dir / 'orders'
satellite_data_dir = data_dir / 'satellites'
outage_request_data_dir = data_dir / 'outages'
ground_station_data_dir = data_dir / 'ground_stations'

# Parse satellites and orders data
two_line_elements = parse_two_line_elements(satellite_data_dir)
satellites = parse_satellites(satellite_data_dir, ts)
jobs = parse_jobs(order_data_dir)
ground_stations = parse_ground_stations(ground_station_data_dir)

sat_name_to_sat: Dict[str, EarthSatellite] = {
    cast(str, sat.name): sat for sat in satellites
}

# Ingest all Batch 1 orders and all maintenance orders except for OrbitMaintenance6 and OrbitMaintenance7
batch1_data_dir = day_in_the_life_data_dir / 'Batch1' / 'JSON'
batch1_jobs = parse_jobs(batch1_data_dir, name_prefix='Batch 1 ')
batch1_outage_requests = parse_maintenance_orders(day_in_the_life_data_dir / 'MaintenanceOrders', satellites)
output1 = run(ScheduleParameters(
    input_hash=None,
    two_line_elements=two_line_elements,
    jobs=batch1_jobs,
    ground_stations=ground_stations,
    outage_requests=batch1_outage_requests,
    ground_station_outage_requests=[]
))
print_api_result(output1, style=Fore.RED)

# Ingest all Batch 2 orders
batch2_data_dir = day_in_the_life_data_dir / 'Batch2' / 'JSON'
batch2_jobs = parse_jobs(batch2_data_dir, name_prefix='Batch 2 ')
output2 = rerun_full(ScheduleParameters(
    input_hash=output1.input_hash,
    two_line_elements=[],
    jobs=batch2_jobs,
    ground_stations=[],
    outage_requests=[],
    ground_station_outage_requests=[]
))
print_api_result(output2, style='\033[38;2;255;165;0m')

# Ingest all Batch 3 orders
batch3_data_dir = day_in_the_life_data_dir / 'Batch3' / 'JSON'
batch3_jobs = parse_jobs(batch3_data_dir, name_prefix='Batch 3 ')
output3 = rerun_full(ScheduleParameters(
    input_hash=output2.input_hash,
    two_line_elements=[],
    jobs=batch3_jobs,
    ground_stations=[],
    outage_requests=[],
    ground_station_outage_requests=[]
))
print_api_result(output3, style=Fore.YELLOW)

# Ingest outage for GATN and SOSO-3
outage_request_soso3 = OutageRequest(
    name='Outage Request SOSO-3',
    satellite_name='SOSO-3',
    start=datetime.datetime.fromisoformat('2023-10-02T10:00:00'),
    end=datetime.datetime.fromisoformat('2023-10-02T15:00:00')
)
outage_request_soso3.assign_satellite(sat_name_to_sat)

ground_station_outage_request_gatineau = GroundStationOutageRequest(
    name='Ground Station Outage Request GATN',
    ground_station=[g for g in ground_stations if g.name == 'Gatineau'][0],
    start=datetime.datetime.fromisoformat('2023-10-02T02:00:00'),
    end=datetime.datetime.fromisoformat('2023-10-02T08:00:00')
)

output3_1 = rerun_full(ScheduleParameters(
    input_hash=output3.input_hash,
    two_line_elements=[],
    jobs=[],
    ground_stations=[],
    outage_requests=[outage_request_soso3],
    ground_station_outage_requests=[ground_station_outage_request_gatineau]
))
print_api_result(output3_1, style=Fore.GREEN)

# Ingest all Batch 4 orders
batch4_data_dir = day_in_the_life_data_dir / 'Batch4' / 'JSON'
batch4_jobs = parse_jobs(batch4_data_dir, name_prefix='Batch 4 ')
output4 = rerun_full(ScheduleParameters(
    input_hash=output3_1.input_hash,
    two_line_elements=[],
    jobs=batch4_jobs,
    ground_stations=[],
    outage_requests=[],
    ground_station_outage_requests=[]
))
print_api_result(output4, style=Fore.CYAN)

# Ingest all Batch 5 orders
batch5_data_dir = day_in_the_life_data_dir / 'Batch5' / 'JSON'
batch5_jobs = parse_jobs(batch5_data_dir, name_prefix='Batch 5 ')
output5 = rerun_full(ScheduleParameters(
    input_hash=output4.input_hash,
    two_line_elements=[],
    jobs=batch5_jobs,
    ground_stations=[],
    outage_requests=[],
    ground_station_outage_requests=[]
))
print_api_result(output5, style='\033[38;2;75;0;130m')

# Ingest OrbitMaintenance6 and OrbitMaintenance7
extra_maintenance_orders = parse_maintenance_orders(day_in_the_life_data_dir / 'MaintenanceOrders2', satellites)
output5_1 = rerun_full(ScheduleParameters(
    input_hash=output5.input_hash,
    two_line_elements=[],
    jobs=batch5_jobs,
    ground_stations=[],
    outage_requests=extra_maintenance_orders,
    ground_station_outage_requests=[]
))
print_api_result(output5_1, style=Fore.MAGENTA)






# output = run(ScheduleParameters(
#     two_line_elements=two_line_elements,
#     jobs=batch1_jobs,
#     ground_stations=ground_stations,
#     outage_requests=[]
#     # outage_requests=[
#     #     OutageRequest(name="hi1", satellite_name="SOSO-1", start=datetime.datetime(year=2023, month=1, day=1), end=datetime.datetime(year=2024, month=1, day=1)),
#     #     OutageRequest(name="hi2", satellite_name="SOSO-2", start=datetime.datetime(year=2023, month=1, day=1), end=datetime.datetime(year=2024, month=1, day=1)),
#     #     OutageRequest(name="hi3", satellite_name="SOSO-3", start=datetime.datetime(year=2023, month=1, day=1), end=datetime.datetime(year=2024, month=1, day=1)),
#     #     OutageRequest(name="hi4", satellite_name="SOSO-4", start=datetime.datetime(year=2023, month=1, day=1), end=datetime.datetime(year=2024, month=1, day=1)),
#     #     OutageRequest(name="hi5", satellite_name="SOSO-5", start=datetime.datetime(year=2023, month=1, day=1), end=datetime.datetime(year=2024, month=1, day=1))
#     # ]
# ))
