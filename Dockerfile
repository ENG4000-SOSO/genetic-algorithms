# Start with a slim Python 3.10 image. Slim was chosen over Alpine to remove the
# need for any apk installs.
FROM python:3.10-slim

# Set working directory to /app.
WORKDIR /app

# Copy setup.py. This is what defines this project as a Python package,
# including its dependencies.
COPY ./setup.py /app/setup.py

# Copy pyproject.toml. It tells Pip which build system to use. This is required
# along with setup.py to make this project a full Python package.
COPY ./pyproject.toml /app/pyproject.toml

# Copy the logging configuration. This defines all the loggers in their project,
# along with their configuration, in a single file.
COPY ./logging_config.ini /app/logging_config.ini

# Copy DE421 ephemeris.
COPY ./de421.bsp /app/de421.bsp

# Copy the source code.
COPY ./src /app/src

# Install dependencies. Since this is a proper Python package with setup.py and
# pyproject.toml files, we can use pip install without the requirements.txt.
RUN pip install --no-cache-dir --upgrade .

# Start the FastAPI server with Uvicorn.
CMD ["uvicorn", "soso.api:app", "--host", "0.0.0.0", "--port", "8080"]
