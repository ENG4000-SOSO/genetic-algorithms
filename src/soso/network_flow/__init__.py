'''
Top-level package for the network flow aspect of the SOSO project.
'''


from .edge_types import \
    Edges, \
    GraphEdge, \
    JobToSatelliteTimeSlotEdge, \
    SatelliteTimeSlot, \
    SatelliteTimeSlotToSinkEdge, \
    SourceToJobEdge
from .network_flow_scheduler_improved import run_network_flow
