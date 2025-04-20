'''
Definitions of FastAPI HTTP endpoints.
'''


import logging
import logging.config
logging.config.fileConfig('logging_config.ini')

from fastapi import FastAPI, Request

from soso.api.api_utils import get_client_ip_and_port
from soso.interface import run, ScheduleParameters, ScheduleOutput


logger: logging.Logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/schedule", response_model=ScheduleOutput)
def schedule(params: ScheduleParameters, request: Request) -> ScheduleOutput:
    client_host, client_port = get_client_ip_and_port(request)
    logger.info(f'Received scheduling request from {client_host}:{client_port}')
    try:
        result = run(params)
        return result
    except Exception as e:
        logger.error(e)
        raise e


@app.get("/ping")
def ping(request: Request):
    client_host, client_port = get_client_ip_and_port(request)
    logger.info(f'Received ping request from {client_host}:{client_port}')
    return {"message": "pong"}


if __name__ == "__main__":
    raise Exception(
        '\n\nPlease run this app with Uvicorn instead of executing this file. '
        'The command will be something like this:\n\n'
        '\tuvicorn soso.api:app --host 0.0.0.0 --port 8080\n\n'
        'Just make sure you have Uvicorn installed (it should be installed if '
        'you execute\n\n'
        '\tpip install .\n\n'
        'in this project\'s base directory).'
    )
