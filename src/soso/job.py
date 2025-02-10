'''
Definitions of the job class (jobs that satellites can be asked to perform) and
additional data types to facilitate its functionality.
'''


from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from intervaltree import Interval


class Priority(Enum):
    '''
    The priority of a job.
    '''

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class SatellitePassLocation:
    name: str
    latitude: float
    longitude: float

    def __init__(
        self,
        name: str,
        latitude: float,
        longitude: float
    ):
        if not name:
            raise Exception('name missing')
        if not latitude:
            raise Exception('latitude missing')
        if not longitude:
            raise Exception('longitude missing')

        self.name = name
        self.latitude = latitude
        self.longitude = longitude


class Job(SatellitePassLocation):
    '''
    Representation of a job that a satellite can be asked to perform.
    '''

    priority: Priority
    start: datetime
    end: datetime
    delivery: datetime
    size: float = 128_000_000 # TODO: MAKE THIS DYNAMIC

    def __init__(
        self,
        name: str,
        start: str,
        end: str,
        delivery: str,
        priority: str,
        latitude: int,
        longitude: int
    ):
        super().__init__(name, latitude, longitude)

        if not start:
            raise Exception('start time missing')
        if not end:
            raise Exception('end time missing')
        if not delivery:
            raise Exception('delivery time missing')
        if not priority:
            raise Exception('priority missing')

        self.start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        self.end = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        self.delivery = datetime.fromisoformat(delivery).replace(tzinfo=timezone.utc)
        self.priority = Priority(priority)

    def __str__(self):
        return f'{self.name} P{self.priority.value}'

    def __repr__(self) -> str:
        return str(self)

    def interval(self) -> Interval:
        return Interval(self.start, self.end, str(self))


class GroundStation(SatellitePassLocation):
    '''
    Representation of a job that a satellite can be asked to perform.
    '''

    height: float
    mask: int
    uplink_rate: int
    downlink_rate: int

    def __init__(
        self,
        name: str,
        latitude: float,
        longitude: float,
        height: float,
        mask: int,
        uplink_rate: int,
        downlink_rate: int
    ):
        super().__init__(name, latitude, longitude)

        if not height:
            raise Exception('height time missing')
        if not mask:
            raise Exception('mask time missing')
        if not uplink_rate:
            raise Exception('uplink_rate missing')
        if not downlink_rate:
            raise Exception('downlink_rate missing')

        self.height = height
        self.mask = mask
        self.uplink_rate = uplink_rate
        self.downlink_rate = downlink_rate

    def __str__(self):
        return f'{self.name} at lat: {self.latitude}, lon: {self.longitude}, height: {self.height}'

    def __repr__(self) -> str:
        return str(self)
