'''
Types used in the interval trees for the SOSO algorithm.
'''


from dataclasses import dataclass
import datetime
from typing import List

from soso.job import Job


@dataclass
class SatelliteInterval:
    '''
    Representation of an interval in a satellite that contains begin and end
    times and a list of jobs that can each be scheduled in the satellite within
    the begin and end times.
    '''

    begin: datetime.datetime
    '''
    The beginning time of the interval.
    '''

    end: datetime.datetime
    '''
    The ending time of the interval.
    '''

    data: List[Job]
    '''
    The list of jobs where each job can be scheduled in this interval.
    '''
