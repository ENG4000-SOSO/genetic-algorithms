import os


class ENV_KEY:
    RUN_ENV = 'RUN_ENV'

    AWS_REGION_NAME = 'AWS_REGION_NAME'

    DYNAMODB_TABLE_NAME = 'DYNAMODB_TABLE_NAME'

    JOB_ID = 'JOB_ID'

    S3_BUCKET_NAME = 'S3_BUCKET_NAME'


def get_env(key: str) -> str:
    if not (key in os.environ and len(os.environ[key]) > 0):
        raise Exception(
            f'The environment variable {key} is either not set or is an empty '
                'string'
        )

    return os.environ[key]
