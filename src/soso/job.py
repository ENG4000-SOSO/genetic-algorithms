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


class Job:
    '''
    Representation of a job that a satellite can be asked to perform.
    '''

    name: str
    start: datetime
    end: datetime
    priority: Priority
    latitude: int
    longitude: int

    def __init__(
        self,
        name: str,
        start: Optional[str],
        end: Optional[str],
        priority: Optional[str],
        latitude: int,
        longitude: int
    ):
        if not name:
            raise Exception('name missing')
        if not start:
            raise Exception('start time missing')
        if not end:
            raise Exception('end time missing')
        if not priority:
            raise Exception('priority missing')
        if not latitude:
            raise Exception('latitude missing')
        if not longitude:
            raise Exception('longitude missing')

        self.name = name
        self.start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        self.end = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        self.priority = Priority(priority)
        self.latitude = latitude
        self.longitude = longitude

    def __str__(self):
        return f'{self.name} P{self.priority.value}'

    def __repr__(self) -> str:
        return str(self)

    def interval(self) -> Interval:
        return Interval(self.start, self.end, str(self))
