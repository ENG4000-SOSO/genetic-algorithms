'''
'''


import logging
import logging.config
logging.config.fileConfig('logging_config.ini')

from fastapi import FastAPI

from soso.interface import run, ScheduleParameters, ScheduleOutput


app = FastAPI()


@app.post("/schedule", response_model=ScheduleOutput)
def schedule(params: ScheduleParameters) -> ScheduleOutput:
    try:
        result = run(params)
    except Exception as e:
        print('error',flush=True)
        print(e,flush=True)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
