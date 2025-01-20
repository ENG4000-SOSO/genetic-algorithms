'''
Definitions of the job class (jobs that satellites can be asked to perform) and
additional data types to facilitate its functionality.
'''


from datetime import datetime, timezone
from typing import Optional


# class GroundStation:
#     '''
#     Representation of a job that a satellite can be asked to perform.
#     '''

#     name: str
#     latitude: float
#     longitude: float
#     height: float
#     mask: int
#     uplink_rate: int
#     downlink_rat: int

#     def __init__(
#         self,
#         name: str,
#         latitude: float,
#         longitude: float,
#         height: float,
#         mask: int,
#         uplink_rate: int,
#         downlink_rate: int
#     ):
#         if not name:
#             raise Exception('name missing')
#         if not latitude:
#             raise Exception('latitude missing')
#         if not longitude:
#             raise Exception('longitude missing')
#         if not height:
#             raise Exception('height time missing')
#         if not mask:
#             raise Exception('mask time missing')
#         if not uplink_rate:
#             raise Exception('uplink_rate missing')
#         if not downlink_rate:
#             raise Exception('downlink_rate missing')

#         self.name = name
#         self.latitude = latitude
#         self.longitude = longitude
#         self.height = height
#         self.mask = mask
#         self.uplink_rate = uplink_rate
#         self.downlink_rate = downlink_rate

#     def __str__(self):
#         return f'{self.name} at lat: {self.latitude}, lon: {self.longitude}, height: {self.height}'

#     def __repr__(self) -> str:
#         return str(self)
