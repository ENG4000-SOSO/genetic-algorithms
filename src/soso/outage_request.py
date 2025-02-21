'''
Definition of the outage request class.
'''


from datetime import datetime, timezone
from typing import Dict

from intervaltree import Interval
from pydantic import BaseModel, ConfigDict, field_validator, PrivateAttr
from skyfield.api import EarthSatellite


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
