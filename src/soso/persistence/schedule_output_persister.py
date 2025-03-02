'''
This module contains interfaces and implementations for persisting scheduling
parameters and output between algorithm runs. This allows for the algorithm to
be re-run with the previous run's data.
'''


from abc import ABC, abstractmethod
import json
from pathlib import Path

from pydantic import BaseModel

from soso.interface.interface_types import ScheduleOutput, ScheduleParameters


# Define storage directory and ensure it exists
STORAGE_DIR = Path.cwd() / 'storage'
STORAGE_DIR.mkdir(exist_ok=True)


class SchedulingInputOutputData(BaseModel):
    '''
    Data model holding information about a scheduling run. Includes the input
    parameters and output data, as well as the hash of the input parameters to
    be used for retrieving this data.
    '''

    params_hash: str
    '''
    Hash of the schedule parameters.
    '''

    params: ScheduleParameters
    '''
    The parameters of the scheduling run.
    '''

    output: ScheduleOutput
    '''
    The output of the scheduling run.
    '''


class ScheduleNotFoundException(Exception):
    '''
    Exception raised when a schedule is not found given the input parameters
    hash as the key.
    '''

    def __init__(self, input_hash: str):
        super().__init__(f'Schedule with hash {input_hash} not found')


class SchedulingDataPersister(ABC):
    '''
    Abstract base class for scheduling output persisters.
    '''

    @abstractmethod
    def persist(self, schedule_params: ScheduleParameters, schedule_output: ScheduleOutput) -> None:
        '''
        Persist the scheduling run's input and output.

        Args:
            schedule_params: The input parameters of the scheduling run.

            schedule_output: The output of the scheduling run.
        '''
        pass

    @abstractmethod
    def retrieve(self, params_hash: str) -> SchedulingInputOutputData:
        '''
        Retrieve the schedule output by hash.
        
        Args:
            params_hash: The hash of the scheduling run's parameters.

        Returns:
            The retrieved scheduling run's data.
        '''
        pass


class FilePersister(SchedulingDataPersister):
    '''
    File-based implementation of the scheduling data persister.
    '''

    def persist(self, schedule_params: ScheduleParameters, schedule_output: ScheduleOutput):
        scheduling_data = SchedulingInputOutputData(
            params_hash=schedule_output.input_hash,
            params=schedule_params,
            output=schedule_output
        )

        filepath = STORAGE_DIR / f'{schedule_output.input_hash}.json'

        with open(filepath, 'w') as f:
            # Write the scheduling input and output data (making sure to sort
            # the keys of the JSON object for consistency)
            f.write(
                json.dumps(scheduling_data.model_dump_json(), sort_keys=True)
            )

    def retrieve(self, input_hash: str) -> SchedulingInputOutputData:
        filepath = STORAGE_DIR / f'{input_hash}.json'
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
                return SchedulingInputOutputData.model_validate_json(data)
        raise ScheduleNotFoundException(input_hash)


def get_persister() -> SchedulingDataPersister:
    return FilePersister()
