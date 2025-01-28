'''
Class (and type) definitions of entities used specifically for the network flow
optimization solution to the satellite scheduling problem.
'''


from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast, Dict, List, NamedTuple, TypeAlias, TypeVar

from skyfield.api import EarthSatellite

from soso.job import GroundStation, Job


X = TypeVar("X")

SatelliteToList: TypeAlias = Dict[EarthSatellite, List[X]]
'''
Type alias representing a mapping (dictionary) of satellites to lists of generic
parameter `X`.
'''


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


@dataclass(frozen=True)
class RateLimiter:
    '''
    Representation of a rate limiter node.

    'Rate limiter' nodes were added to the network flow graph to prevent
    satellite timeslots from mapping to multiple ground station passes. This
    situation is prevented by including the rate limiters as a one-to-one
    mapping from satellite timeslots to themselves.

    By including these rate limiters, we restrict the flow coming out of
    satellite timeslot nodes, thereby restricting the number of ground station
    passes connected to a satellite timeslot.
    '''

    satellite_timeslot: SatelliteTimeSlot
    '''
    The satellite timeslot node that the rate limiter is being applied to.
    '''

    @property
    def satellite(self) -> EarthSatellite:
        '''
        The satellite for which this rate limiter's time slot is represented
        for.
        '''
        return self.satellite_timeslot.satellite

    @property
    def start(self) -> datetime:
        '''
        The start time of this rate limiter's time slot.
        '''
        return self.satellite_timeslot.start

    @property
    def end(self) -> datetime:
        '''
        The end time of this rate limiter's time slot.
        '''
        return self.satellite_timeslot.end


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



class SatelliteTimeSlotToRateLimiter(GraphEdge):
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
    def rate_limiter(self) -> RateLimiter:
        '''
        The ground station node of the edge.
        '''
        return cast(RateLimiter, self.v)

    @property
    def flow(self) -> int:
        '''
        The flow across the edge.
        '''
        return self.f


class RateLimiterEdge(GraphEdge):
    '''
    Representation of a graph edge connecting a job node to a satellite time
    slot node.
    '''

    @property
    def rate_limiter(self) -> RateLimiter:
        '''
        '''
        return cast(RateLimiter, self.u)

    @property
    def ground_station(self) -> GroundStationPassTimeSlot:
        '''
        The satellite time slot node of the edge. This will be the string
        representation of an interval (with a start and end datetime) for a
        satellite.
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

    jobToSatelliteTimeSlotEdges: SatelliteToList[JobToSatelliteTimeSlotEdge]
    '''
    A dictionary that maps each satellite to a list of edges. The edges are from
    jobs to time slots for the satellite.
    '''

    satelliteTimeSlotToRateLimiterEdges: SatelliteToList[SatelliteTimeSlotToRateLimiter]
    '''
    A dictionary that maps each satellite to a list of edges. The edges are from
    time slots for that satellite to rate limiter nodes that restrict the flow
    to ground stations.
    '''

    rateLimiterToGroundStationEdges: SatelliteToList[RateLimiterEdge]
    '''
    A dictionary that maps each satellite to a list of edges. The edges are from
    rate limiter nodes to ground station pass timeslots for the satellite.
    '''

    groundStationPassToSinkEdge: SatelliteToList[GroundStationPassToSinkEdge]
    '''
    A dictionary that maps each satellite to a list of edges. The edges are from
    ground station passes for that satellite to the sink node.
    '''


@dataclass
class SolutionTimeSlot:
    '''
    A unit of a network flow solution, containing a job, a satellite timeslot
    for the job to be scheduled in, and a ground station timeslot for the job to
    be downlinked in.
    '''

    job: Job
    '''
    The job being scheduled.
    '''

    satelliteTimeSlot: SatelliteTimeSlot
    '''
    The satellite timeslot that the job will be scheduled in.
    '''

    groundStationPassTimeSlot: GroundStationPassTimeSlot
    '''
    The timeslot for the ground station pass where the job will be downlinked.
    '''


@dataclass
class NetworkFlowSolution:
    '''
    A representation of the full network flow solution.
    '''

    solutionTimeSlots: SatelliteToList[SolutionTimeSlot]
    '''
    A mapping of each satellite to a list of timeslots containing jobs and
    information on when those jobs will be performed and downlinked.
    '''

    satelliteTimeSlots: SatelliteToList[JobToSatelliteTimeSlotEdge]
    '''
    A mapping of each satellite to a list of edges from the optimal network flow
    graph, where the edges connect job and satellite timeslot nodes.
    '''

    groundStationPassTimeSlots: SatelliteToList[GroundStationPassToSinkEdge]
    '''
    A mapping of each satellite to a list of edges from the optimal network flow
    graph, where the edges connect a ground station pass timeslot to the sink
    node.
    '''
