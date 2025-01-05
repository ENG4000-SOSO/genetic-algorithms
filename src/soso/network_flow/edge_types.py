'''
Class (and type) definitions of entities used specifically for the network flow
optimization solution to the satellite scheduling problem.
'''


from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast, List, Dict, NamedTuple

from skyfield.api import EarthSatellite

from soso.job import Job


@dataclass(frozen=True)
class SatelliteTimeSlot:
    satellite: EarthSatellite
    start: datetime
    end: datetime

    def __hash__(self):
        return hash((self.satellite, self.start, self.end))


class GraphEdge(NamedTuple):
    '''
    Representation of an edge in a network flow graph.
    '''

    u: Any
    '''
    The first node of the edge.
    '''

    v: Any
    '''
    The second node of the edge.
    '''

    f: int
    '''
    The flow across the edge.
    '''


class SourceToJobEdge(GraphEdge):
    '''
    Representation of a graph edge connecting a source node to a job node.
    '''

    @property
    def source(self) -> str:
        '''
        The source node of the edge. This should always be the string
        literal 'source' because the source node never changes.
        '''
        return cast(str, self.u)

    @property
    def job(self) -> Job:
        '''
        The job node of the edge. This will be the string representation of the
        job.
        '''
        return cast(Job, self.v)

    @property
    def flow(self) -> int:
        '''
        The flow across the edge.
        '''
        return self.f


class JobToSatelliteTimeSlotEdge(GraphEdge):
    '''
    Representation of a graph edge connecting a job node to a satellite time
    slot node.
    '''

    @property
    def job(self) -> Job:
        '''
        The job node of the edge. This will be the string representation of the
        job.
        '''
        return cast(Job, self.u)

    @property
    def satellite_timeslot(self) -> SatelliteTimeSlot:
        '''
        The satellite time slot node of the edge. This will be the string
        representation of an interval (with a start and end datetime) for a
        satellite.
        '''
        return cast(SatelliteTimeSlot, self.v)

    @property
    def flow(self) -> int:
        '''
        The flow across the edge.
        '''
        return self.f


class SatelliteTimeSlotToSinkEdge(GraphEdge):
    '''
    Representation of a graph edge connecting a job node to a satellite time
    slot node.
    '''

    @property
    def satellite_timeslot(self) -> SatelliteTimeSlot:
        '''
        The satellite time slot node of the edge. This will be the string
        representation of an interval (with a start and end datetime) for a
        satellite.
        '''
        return cast(SatelliteTimeSlot, self.u)

    @property
    def sink(self) -> str:
        '''
        The sink node of the edge. This should always be the string literal
        'sink' because the sink node never changes.
        '''
        return cast(str, self.v)

    @property
    def flow(self) -> int:
        '''
        The flow across the edge.
        '''
        return self.f


@dataclass(frozen=True)
class Edges:
    '''
    Representation of a set of edges in a network flow graph that models the
    job-satellite scheduling optimization problem.
    '''

    sourceToJobEdges: List[SourceToJobEdge]
    '''
    List of edges from the source node to each job.
    '''

    jobToSatelliteTimeSlotEdges: Dict[EarthSatellite, List[JobToSatelliteTimeSlotEdge]]
    '''
    A dictionary that maps each satellite to a list of edges. The edges are from
    jobs to time slots for the satellite.
    '''

    satelliteTimeSlotToSinkEdges: Dict[EarthSatellite, List[SatelliteTimeSlotToSinkEdge]]
    '''
    A dictionary that maps each satellite to a list of edges. The edges are from
    time slots for that satellite to the sink node.
    '''
