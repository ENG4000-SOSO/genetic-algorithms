'''
Functionality to run the SOSO scheduler on AWS.
'''


from datetime import datetime
import json
import logging
from typing import cast

import boto3
from mypy_boto3_dynamodb.service_resource import DynamoDBServiceResource
from mypy_boto3_s3.client import S3Client

from soso.interface.interface_types import ScheduleParameters
from soso.interface import run
from soso.env import ENV_KEY, get_env


AWS_LOGGER_QUAL_NAME = 'soso.aws'
logger: logging.Logger = logging.getLogger(AWS_LOGGER_QUAL_NAME)


def get_parameters_from_s3(
    aws_region_name: str,
    s3_bucket_name: str,
    dynamodb_table_name: str,
    job_id: str
) -> ScheduleParameters:
    logger.info(f'Getting item from DynamoDb table {dynamodb_table_name} with job_id {job_id} in region {aws_region_name}')

    dynamodb: DynamoDBServiceResource = cast(
        DynamoDBServiceResource,
        boto3.resource('dynamodb', region_name=aws_region_name)
    )

    table = dynamodb.Table(dynamodb_table_name)
    response = table.get_item(Key={'job_id': job_id})
    job_metadata = response.get('Item')

    if job_metadata is None:
        raise Exception(f'Job metadata with id {job_id} does not exist')

    bucket = s3_bucket_name
    key = job_metadata['input_object_key']

    logger.info(f'Getting {key} from S3 bucket {bucket}')

    s3: S3Client = cast(S3Client, boto3.client('s3'))
    response = s3.get_object(Bucket=bucket, Key=str(key))

    json_data = json.loads(response['Body'].read())

    params = ScheduleParameters.model_validate_json(json_data)

    logger.info(f'Updating status of {job_id} in DynamoDb table {dynamodb_table_name} to "pending"')

    table.update_item(
        Key={'job_id': job_id},
        UpdateExpression='SET #st = :s, updated_at = :u',
        ExpressionAttributeNames={
            '#st': 'status',
        },
        ExpressionAttributeValues={
            ':s': 'pending',
            ':u': datetime.now().isoformat(),
        }
    )

    return params


def run_aws():
    if get_env(ENV_KEY.RUN_ENV) != 'AWS':
        raise Exception(
            'The AWS interface can only be used when the "RUN_ENV" environment '
                'variable is set to "AWS"'
        )

    aws_region_name = get_env(ENV_KEY.AWS_REGION_NAME)
    dynamodb_table_name = get_env(ENV_KEY.DYNAMODB_TABLE_NAME)
    job_id = get_env(ENV_KEY.JOB_ID)
    s3_bucket_name = get_env(ENV_KEY.S3_BUCKET_NAME)

    params = get_parameters_from_s3(
        aws_region_name,
        s3_bucket_name,
        dynamodb_table_name,
        job_id
    )

    run(params)
