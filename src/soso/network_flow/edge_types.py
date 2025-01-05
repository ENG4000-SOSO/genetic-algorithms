'''
Class (and type) definitions of entities used specifically for the network flow
optimization solution to the satellite scheduling problem.
'''


from dataclasses import dataclass
from typing import List, Dict, NamedTuple

from skyfield.api import EarthSatellite


class GraphEdge(NamedTuple):
    '''
    Representation of an edge in a network flow graph.
    '''

    u: str
    '''
    The first node of the edge.
    '''

    v: str
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
        return self.u

    @property
    def job(self) -> str:
        '''
        The job node of the edge. This will be the string representation of the
        job.
        '''
        return self.v

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
    def job(self) -> str:
        '''
        The job node of the edge. This will be the string representation of the
        job.
        '''
        return self.u

    @property
    def satellite_timeslot(self) -> str:
        '''
        The satellite time slot node of the edge. This will be the string
        representation of an interval (with a start and end datetime) for a
        satellite.
        '''
        return self.v

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
    def satellite_timeslot(self) -> str:
        '''
        The satellite time slot node of the edge. This will be the string
        representation of an interval (with a start and end datetime) for a
        satellite.
        '''
        return self.u

    @property
    def sink(self) -> str:
        '''
        The sink node of the edge. This should always be the string literal
        'sink' because the sink node never changes.
        '''
        return self.v

    @property
    def flow(self) -> int:
        '''
        The flow across the edge.
        '''
        return self.f


@dataclass
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
