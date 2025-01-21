'''
Class (and type) definitions of entities used specifically for the network flow
optimization solution to the satellite scheduling problem.
'''


from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast, List, Dict, NamedTuple

from skyfield.api import EarthSatellite

from soso.job import GroundStation, Job


@dataclass(frozen=True)
class SatelliteTimeSlot:
    '''
    Representation of a satellite time slot, which contains a satellite, a start
    time, and an end time.
    '''

    satellite: EarthSatellite
    '''
    The satellite for which the time slot is represented for.
    '''

    start: datetime
    '''
    The start time of the time slot.
    '''

    end: datetime
    '''
    The end time of the time slot.
    '''

    def __hash__(self):
        '''
        Hash method is overridden so that satellite time slots with the same
        properties are equal.
        '''

        return hash((self.satellite, self.start, self.end))


@dataclass(frozen=True)
class GroundStationPassTimeSlot:
    '''
    Representation of a time slot where a satellite passes over a ground
    station. This contains a satellite, ground station, start time, and end
    time.
    '''

    satellite: EarthSatellite
    '''
    The satellite that is passing over the ground station in this time slot.
    '''

    ground_station: GroundStation
    '''
    The ground station being passed over in this time slot.
    '''

    start: datetime
    '''
    The start time of the time slot.
    '''

    end: datetime
    '''
    The end time of the time slot.
    '''

    def __hash__(self):
        '''
        Hash method is overridden so that ground station pass time slots with
        the same properties are equal.
        '''

        return hash((self.satellite, self.ground_station, self.start, self.end))


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


class SatelliteTimeSlotToGroundStationPassEdge(GraphEdge):
    '''
    Representation of a graph edge connecting a satellite time slot node to a
    ground station pass node.
    '''

    @property
    def satellite_timeslot(self) -> SatelliteTimeSlot:
        '''
        The satellite time slot node of the edge.
        '''
        return cast(SatelliteTimeSlot, self.u)

    @property
    def ground_station(self) -> GroundStationPassTimeSlot:
        '''
        The ground station node of the edge.
        '''
        return cast(GroundStationPassTimeSlot, self.v)

    @property
    def flow(self) -> int:
        '''
        The flow across the edge.
        '''
        return self.f


class GroundStationPassToSinkEdge(GraphEdge):
    '''
    Representation of a graph edge connecting a ground station pass node to a
    sink node.
    '''

    @property
    def ground_station(self) -> GroundStationPassTimeSlot:
        '''
        The ground station node of the edge.
        '''
        return cast(GroundStationPassTimeSlot, self.u)

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

    satelliteTimeSlotToGroundStationPassEdge: Dict[EarthSatellite, List[SatelliteTimeSlotToGroundStationPassEdge]]
    '''
    A dictionary that maps each satellite to a list of edges. The edges are from
    time slots for that satellite to ground station passes for that satellite.
    '''

    groundStationPassToSinkEdge: Dict[EarthSatellite, List[GroundStationPassToSinkEdge]]
    '''
    A dictionary that maps each satellite to a list of edges. The edges are from
    ground station passes for that satellite to the sink node.
    '''
