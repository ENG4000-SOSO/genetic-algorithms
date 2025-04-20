'''
Module defining the types used to interface with the scheduling algorithm at a
high-level.
'''


from datetime import datetime
import hashlib
import json
from typing import Dict, List, Optional

from pydantic import BaseModel

from soso.job import Job, GroundStation, TwoLineElement
from soso.outage_request import GroundStationOutageRequest, OutageRequest


class ScheduleParameters(BaseModel):
    '''
    Representation of the parameters of the scheduling algorithm.
    '''

    input_hash: Optional[str]
    '''
    Hash of the input parameters of the previous scheduling run. Use this
    attribute only when re-scheduling, otherwise leave this as `None`.
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

    ground_station_outage_requests: List[GroundStationOutageRequest]
    '''
    The outage requests that add constraints to when ground stations are
    unavailable.
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

    input_hash: str
    '''
    The hash of the input parameters that produced this output. Both the user
    and the scheduler should store this value to reference this scheduling run
    in the future in the case of rescheduling.
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

    undownlinkable_orders: List[Job]
    '''
    Orders that were impossible to be downlinked.
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

    @classmethod
    def convert_to_hash(cls, value: ScheduleParameters):
        '''
        Hashes schedule parameters to be used as a key.
        '''

        model_json_string = json.dumps(value.model_dump_json(), sort_keys=True)
        model_json_bytes = model_json_string.encode()
        model_hash = hashlib.sha256(model_json_bytes).hexdigest()
        return model_hash
