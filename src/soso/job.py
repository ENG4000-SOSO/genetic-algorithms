'''
Definitions of the job class (jobs that satellites can be asked to perform) and
additional data types to facilitate its functionality.
'''


from datetime import datetime, timezone
from enum import Enum

from intervaltree import Interval
from pydantic import BaseModel, ConfigDict, field_validator


class Priority(Enum):
    '''
    The priority of a job.
    '''

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class SatellitePassLocation(BaseModel):
    '''
    Representation of a location the a satellite passes over.
    '''

    model_config = ConfigDict(frozen=True)

    name: str
    '''
    The name of the location.
    '''

    latitude: float
    '''
    The latitude of the location.
    '''

    longitude: float
    '''
    The longitude of the location.
    '''


class Job(SatellitePassLocation):
    '''
    Representation of a job that a satellite can be asked to perform.
    '''

    model_config = ConfigDict(
        frozen=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    priority: Priority
    '''
    The priority of the job.
    '''

    start: datetime
    '''
    The start time of the interval in which the job must be performed.
    '''

    end: datetime
    '''
    The end time of the interval in which the job must be performed.
    '''

    delivery: datetime
    '''
    The time by which the job must be delivered to a ground station.
    '''

    size: float = 128_000_000 # TODO: MAKE THIS DYNAMIC
    '''
    The size of the image in bytes.
    '''

    @field_validator('start', 'end', 'delivery', mode='after')
    @classmethod
    def ensure_start_utc(cls, v: datetime) -> datetime:
        '''
        Ensures that the start, end, and delivery times have timezone
        information.
        '''
        return v.replace(tzinfo=timezone.utc)

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

    model_config = ConfigDict(frozen=True)

    height: float
    '''
    The elevation of the ground station.
    '''

    mask: int
    '''
    The mask of the ground station.
    '''

    uplink_rate: int
    '''
    The uplink rate of the ground station in Mbps.
    '''

    downlink_rate: int
    '''
    The downlink rate of the ground station in Mbps.
    '''

    def __str__(self):
        return f'{self.name} at lat: {self.latitude}, lon: {self.longitude}, height: {self.height}'

    def __repr__(self) -> str:
        return str(self)


class TwoLineElement(BaseModel):
    '''
    Representation of a two-line element.
    '''

    model_config = ConfigDict(frozen=True)

    name: str
    '''
    The name of the satellite represented by the two-line element.
    '''

    line1: str
    '''
    The first line of the two-line element.
    '''

    line2: str
    '''
    The second line of the two-line element.
    '''
