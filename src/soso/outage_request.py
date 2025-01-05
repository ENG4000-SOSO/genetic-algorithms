'''
Definition of the outage request class.
'''


from datetime import datetime, timezone
from typing import Optional
from intervaltree import Interval

from skyfield.api import EarthSatellite


class OutageRequest:
    '''
    Representation of an outage request that renders a satellite unusable for a
    certain amount of time.
    '''

    name: str
    satellite: EarthSatellite
    start: datetime
    end: datetime

    def __init__(
        self,
        name: str,
        satellite: EarthSatellite,
        start: Optional[str],
        end: Optional[str]
    ):
        if not name:
            raise Exception('name missing')
        if not satellite:
            raise Exception('satellite missing')
        if not start:
            raise Exception('start time missing')
        if not end:
            raise Exception('end time missing')

        self.name = name
        self.satellite = satellite
        self.start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        self.end = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    def __str__(self):
        return f'{self.name} - {self.satellite.name}'

    def __repr__(self) -> str:
        return str(self)

    def interval(self) -> Interval:
        return Interval(self.start, self.end, str(self))
