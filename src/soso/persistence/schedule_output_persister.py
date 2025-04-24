'''
This module contains interfaces and implementations for persisting scheduling
parameters and output between algorithm runs. This allows for the algorithm to
be re-run with the previous run's data.
'''


from abc import ABC, abstractmethod
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import cast

import boto3
from botocore.exceptions import ClientError
from mypy_boto3_dynamodb.service_resource import DynamoDBServiceResource
from mypy_boto3_s3.client import S3Client
from pydantic import BaseModel

from soso.interface.interface_types import ScheduleOutput, ScheduleParameters
from soso.env import ENV_KEY, get_env


logger: logging.Logger = logging.getLogger(__name__)


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


class S3Persister(SchedulingDataPersister):
    '''
    S3-based implementation of the scheduling data persister.
    '''

    def __init__(
        self,
        aws_region_name: str,
        s3_bucket_name: str,
        dynamodb_table_name: str,
        job_id: str,
        prefix: str = 'output/'
    ):
        self.s3 = cast(S3Client, boto3.client('s3'))
        self.dynamodb = cast(
            DynamoDBServiceResource,
            boto3.resource('dynamodb', region_name=aws_region_name)
        )
        self.s3_bucket_name = s3_bucket_name
        self.dynamodb_table_name = dynamodb_table_name
        self.job_id = job_id
        self.prefix = prefix.rstrip('/') + '/'

    def _get_object_key(self, input_hash: str) -> str:
        return f'{self.prefix}{input_hash}.json'

    def persist(self, schedule_params: ScheduleParameters, schedule_output: ScheduleOutput):
        scheduling_data = SchedulingInputOutputData(
            params_hash=schedule_output.input_hash,
            params=schedule_params,
            output=schedule_output
        )

        object_key = self._get_object_key(schedule_output.input_hash)

        logger.info(f'Updating {self.job_id} in DynamoDb table {self.dynamodb_table_name}')

        table = self.dynamodb.Table(self.dynamodb_table_name)
        table.update_item(
            Key={'job_id': self.job_id},
            UpdateExpression='SET #st = :s, updated_at = :u, output_object_key = :o',
            ExpressionAttributeNames={
                '#st': 'status',
            },
            ExpressionAttributeValues={
                ':s': 'completed',
                ':u': datetime.now().isoformat(),
                ':o': object_key
            }
        )

        logger.info(f'Persisting {object_key} to S3 bucket {self.s3_bucket_name}')

        json_data = json.dumps(scheduling_data.model_dump_json(), sort_keys=True)

        self.s3.put_object(
            Bucket=self.s3_bucket_name,
            Key=object_key,
            Body=json_data,
            ContentType='application/json'
        )

    def retrieve(self, input_hash: str) -> SchedulingInputOutputData:
        object_key = self._get_object_key(input_hash)
        logger.info(f'Retrieving {object_key} from S3 bucket {self.s3_bucket_name}')
        try:
            response = self.s3.get_object(Bucket=self.s3_bucket_name, Key=object_key)
            data = json.loads(response['Body'].read())
            return SchedulingInputOutputData.model_validate_json(data)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise ScheduleNotFoundException(input_hash)
            else:
                raise Exception(e)


def get_persister() -> SchedulingDataPersister:
    try:
        if get_env(ENV_KEY.RUN_ENV) == 'AWS':
            aws_region_name = get_env(ENV_KEY.AWS_REGION_NAME)
            dynamodb_table_name = get_env(ENV_KEY.DYNAMODB_TABLE_NAME)
            bucket = get_env(ENV_KEY.S3_BUCKET_NAME)
            job_id = get_env(ENV_KEY.JOB_ID)

            logger.info(
                f'Using AWS S3 persister for S3 bucket {bucket}, DynamoDb '
                    f'table {dynamodb_table_name} and job ID: {job_id}'
            )

            return S3Persister(
                aws_region_name,
                bucket,
                dynamodb_table_name,
                job_id
            )
    except:
        pass

    logger.info("Using file persister")
    return FilePersister()
