# Satellite Operations Services Optimizer (SOSO) Scheduler

The SOSO scheduler is a satellite scheduling engine designed to optimize the allocation of imaging jobs to satellites and the downlink of data to ground stations. It supports complex constraints such as outages, maintenance, and ground station availability, and is built for extensibility and cloud deployment.

## Features

- **Genetic Algorithm Optimization:** Uses a genetic algorithm to optimize scheduling based on configurable objectives.

- **Network Flow & Bin Packing:** Employs network flow and bin packing algorithms for efficient resource allocation.

- **API Service:** Exposes a REST API for scheduling requests via FastAPI.

- **AWS Integration:** Supports persistence to AWS S3 and DynamoDB for scalable, cloud-based operation.

- **Flexible Input:** Accepts jobs, satellite TLEs, ground stations, and outage requests as input.

- **Rich Output:** Provides detailed scheduling results, including reasons for unscheduled jobs.

## Project Structure

```
src/
  soso/
    api/                # FastAPI wrapper
    bin_packing/        # Bin packing algorithm
    genetic/            # Genetic algorithm
    interface/          # User-facing scheduling interface
    network_flow/       # Network flow algorithm
    persistence/        # Persistence to file or AWS
    utils/              # Utilities for parsing, printing, etc.
    job.py              # Job and resource definitions
    outage_request.py   # Outage and maintenance request definitions

docs/                   # Sphinx documentation code

data/                   # Data to be used as input to the scheduler

day_in_the_life_data/   # Day in the life dataset to be used as input
                        # to the scheduler

main.py               # CLI entry point
day_in_the_life.py    # Scenario driver script
logging_config.ini    # Logger definitions
Dockerfile            # Containerization
Dockerfile.aws        # Containerization for AWS ECR
build_aws.sh          # AWS image build script
run_aws.sh            # AWS container run script
requirements.txt      # Python dependencies
setup.py              # Python package definition
pyproject.toml        # Python package build information
```

## Getting Started

This section specifies how to install, build, and run the scheduler.

### Prerequisites

- Python 3.10+ (version 3.11.11 was used in development)
- [Docker](https://www.docker.com/) (for containerized deployment)
- AWS CLI (for AWS integration)

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/ENG4000-SOSO/genetic-algorithms.git scheduler
    cd scheduler
    ```

2. Install dependencies:

    If you are running locally via the `main.py` CLI entrypoint:

    ```sh
    pip install -r requirements.txt
    ```

    If you are running as a standalone server:

    ```sh
    pip install -e .
    ```

    (Note that the `-e` is so that the project remains editable after installing. If you don't intend to edit it, you can leave out the `-e`.)

    If you are running only as a Docker container, then you don't need to install dependencies. The section ["API Server in a Docker Container"](#api-server-in-a-docker-container) details the necessary instructions.

### Running the Scheduler

There are five main ways to run the scheduler:

| Method                                | Useful for...                                      |
|---------------------------------------|----------------------------------------------------|
| CLI entry point  `main.py`            | quick feedback and debugging                       |
| CLI entry point  `day_in_the_life.py` | running on a large dataset                         |
| API server on the host machine        | debugging the entire server                        |
| API Server in a Docker Container      | running the server without installing dependencies |
| AWS entry point                       | simulating the AWS production environment          |

Below are the instructions for each method.

#### Command Line via `main.py`

Runs a scheduling scenario. The data will be parsed from the JSON files in the [`data/`](./data/) directory.

```sh
python main.py <alg_type>
```

The possible values for `<alg_type>` are:

- `network` - Only runs the network flow algorithm to generate the schedule.

- `bin` - Runs the network flow and bin packing algorithms to generate the schedule.

- `genetic` - Runs the genetic algorithm (which uses both the network flow and bin packing algorithms internally) to generate the schedule.

- `api-full` - Runs the entire scheduler end-to-end by starting it as a server and sending an HTTP request to begin the algorithm.

- `api-request` - Runs the entire scheduler by sending an HTTP request to begin the algorithm. Note that this assumes the scheduler is already running as a server (on port 8000 by default).

#### Command Line via `day_in_the_life.py`

Runs the "day in the life" scheduling scenario that was provided by the Canadian Space Agency. The data will be parsed from the JSON files in the [`day_in_the_life_data/`](./day_in_the_life_data/) directory.

```sh
python day_in_the_life.py
```

#### API Server on Host Machine

Start the FastAPI server:

```sh
uvicorn soso.api:app --reload --app-dir src --host 0.0.0.0 --port 8000
```

Send a scheduling request to `POST /schedule` with the required parameters.

#### API Server in a Docker Container

Build the Docker image:

```sh
docker build -t <name>:<version> .
```

Run as a Docker container (you may need to add some environment variables depending on how you are running the scheduler; see [run_aws.sh](./run_aws.sh) for example):

```sh
docker run -p 8000:8000 <name>:<version>
```

Send a scheduling request to `POST /schedule` with the required parameters.

#### AWS Entry Point

Build and push the Docker image:

```sh
./build_aws.sh <version>
```

Run the scheduler as if it were on AWS (connected to DynamoDb and S3):

```sh
./run_aws.sh <version> <job_id>
```

## Input & Output Formats

- **Format:** As part of the documentation process, Python type hints were given to denote the input and output types of each aspect of the scheduler. The input/output types for the end-to-end algorithm are detailed in [`src/soso/interface/interface_types.py`](./src/soso/interface/interface_types.py).

- **Input:** Jobs, satellites (TLEs), ground stations, and outage requests (see `data/` and `day_in_the_life_data/`).

- **Output:** Schedule results, including planned orders and unscheduled jobs, persisted locally or to AWS.

  If run locally, then, by default, the outputs will be saved to the `./storage` directory (will be created if it does not exist). If run via AWS, they will be saved in an S3 bucket. More details can be found in [the persistence section of the source code](./src/soso/persistence/schedule_output_persister.py).

## Documentation via Sphinx

Full documentation is available in the `docs/` directory. To build the docs:

```sh
cd docs
make html
```

## License

MIT License. See the [`LICENSE`](./LICENSE) file for details.
