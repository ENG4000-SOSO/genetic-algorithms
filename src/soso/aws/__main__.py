'''
Entry point for running the SOSO scheduling engine through AWS.
'''


import logging
import logging.config
logging.config.fileConfig('logging_config.ini')

from . import AWS_LOGGER_QUAL_NAME, run_aws


logger: logging.Logger = logging.getLogger(AWS_LOGGER_QUAL_NAME)


if __name__ == '__main__':
    logger.info(f'Starting SOSO scheduling for AWS')
    run_aws()
