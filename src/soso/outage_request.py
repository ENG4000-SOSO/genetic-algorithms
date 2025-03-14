'''
Definition of the outage request class.
'''


from datetime import datetime, timezone, timedelta
from typing import cast, Dict, List

from intervaltree import Interval
from pydantic import BaseModel, ConfigDict, field_validator, PrivateAttr
from skyfield.api import EarthSatellite

from soso.job import GroundStation


class OutageRequest(BaseModel):
    '''
    Representation of an outage request that renders a satellite unusable for a
    certain amount of time.
    '''

    model_config = ConfigDict(
        frozen=True,
        json_encoders={datetime: lambda v: v.isoformat()},
    )

    name: str
    '''
    The name of the outage request.
    '''

    satellite_name: str
    '''
    The name of the satellite that the outage applies to.
    '''

    start: datetime
    '''
    The start time of the outage.
    '''

    end: datetime
    '''
    The end time of the outage.
    '''

    _satellite: EarthSatellite | None = PrivateAttr(default=None)
    '''
    The satellite object that represents the satellite that the outage request
    applies to.
    '''

    @field_validator('start', 'end', mode='after')
    @classmethod
    def ensure_start_utc(cls, v: datetime) -> datetime:
        '''
        Ensures that the start and end times have timezone information.
        '''
        return v.replace(tzinfo=timezone.utc)

    @property
    def satellite(self) -> EarthSatellite:
        '''
        The satellite that is associated with this outage request.

        This property will cause an exception if the satellite object has not
        yet been assigned to the outage request.
        '''

        if self._satellite is None:
            raise Exception(
                'Satellite object has not yet been assigned for outage '
                    f'request {self.name}'
            )

        return self._satellite

    def assign_satellite(
        self,
        sat_name_to_sat: Dict[str, EarthSatellite]
    ) -> None:
        '''
        Assigns the satellite object to the outage request, based on the
        satellite's name that was given to the outage request.

        The reason we have to do this is because Pydantic can't handle the
        `EarthSatellite` object, so we need to use only the satellite's name,
        and retroactively add the satellite object.

        Args:
            sat_name_to_sat: A dictionary mapping satellite names to satellite
                objects.

        Raises:
            Exception: If the satellite name given to the outage request is not
                found in the dictionary.
        '''

        if self.satellite_name not in sat_name_to_sat:
            raise Exception(f'satellite {self.satellite_name} not found')

        self._satellite = sat_name_to_sat[self.satellite_name]

    def __str__(self):
        return f'{self.name} - {self.satellite_name}'

    def __repr__(self) -> str:
        return str(self)

    def interval(self) -> Interval:
        return Interval(self.start, self.end, str(self))


class GroundStationOutageRequest(BaseModel):
    '''
    Representation of an outage request that renders a ground station unusable
    for a certain amount of time.
    '''

    model_config = ConfigDict(
        frozen=True,
        json_encoders={datetime: lambda v: v.isoformat()},
    )

    name: str
    '''
    The name of the outage request.
    '''

    ground_station: GroundStation
    '''
    The ground station that the outage applies to.
    '''

    start: datetime
    '''
    The start time of the outage.
    '''

    end: datetime
    '''
    The end time of the outage.
    '''

    @field_validator('start', 'end', mode='after')
    @classmethod
    def ensure_start_utc(cls, v: datetime) -> datetime:
        '''
        Ensures that the start and end times have timezone information.
        '''
        return v.replace(tzinfo=timezone.utc)

    def __str__(self):
        return f'{self.name} - {self.ground_station.name}'

    def __repr__(self) -> str:
        return str(self)

    def interval(self) -> Interval:
        return Interval(self.start, self.end, str(self))


class Window(BaseModel):

    model_config = ConfigDict(
        frozen=True,
        json_encoders={datetime: lambda v: v.isoformat()},
    )

    Start: datetime
    End: datetime


class Frequency(BaseModel):

    model_config = ConfigDict(frozen=True)

    minimum_gap: int
    maximum_gap: int


class RepeatCycle(BaseModel):

    model_config = ConfigDict(frozen=True)

    Frequency: Frequency
    Repetition: int


class MaintenanceOrder(BaseModel):

    model_config = ConfigDict(frozen=True)

    Target: str
    Activity: str
    Window: Window
    Duration: int
    RepeatCycle: RepeatCycle
    PayloadOutage: bool


def handle_repeated_outages(maintenance_order: MaintenanceOrder, satellites: List[EarthSatellite]) -> List[OutageRequest]:
    '''
    '''

    sat_name_to_sat: Dict[str, EarthSatellite] = {
        cast(str, sat.name): sat for sat in satellites
    }

    outage_requests: List[OutageRequest] = []
    start_time = maintenance_order.Window.Start
    for i in range(maintenance_order.RepeatCycle.Repetition):
        outage_request = OutageRequest(
            name=f'{maintenance_order.Target}-{maintenance_order.Activity}-{i}',
            satellite_name=maintenance_order.Target,
            start=start_time + timedelta(seconds=maintenance_order.Duration),
            end=maintenance_order.Window.End
        )

        outage_request.assign_satellite(sat_name_to_sat)
        outage_requests.append(outage_request)

        start_time += timedelta(
            seconds=maintenance_order.RepeatCycle.Frequency.minimum_gap
        )

    return outage_requests
