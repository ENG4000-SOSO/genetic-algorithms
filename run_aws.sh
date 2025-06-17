#!/bin/bash

# -----------------------------------------------------------------------------
# This script runs a Docker container of the SOSO scheduler service, configured
# to connect to AWS resources (S3 and DynamoDB). It passes the  required AWS
# credentials and job-specific environment variables to the container.
#
# Usage:
#   ./run_scheduler.sh <version> <job_id>
#
# Arguments:
#   <version>  - Required. The version tag of the scheduler Docker image to
#                  run.
#   <job_id>   - Required. The job ID to process inside the scheduler.
#
#   Behavior:
#     1. Verifies both required arguments are present.
#     2. Sets relevant AWS environment variables.
#     3. Retrieves credentials from AWS CLI config.
#     4. Runs the specified version of the "soso-scheduler" container with the
#          proper environment variables.
#
# Requirements:
#   - AWS CLI configured with valid credentials and region.
#   - Docker installed and running.
#   - The specified version of the scheduler image must exist locally.
# -----------------------------------------------------------------------------

# Check if exactly two arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <version> <job_id>"
    exit 1
fi

# Parse arguments
version=$1
job_id=$2

# Set environment variables

export AWS_REGION_NAME='us-east-1'

export S3_BUCKET_NAME='soso-storage'

export DYNAMODB_TABLE_NAME='soso-schedule-metadata'

# Run the Docker container with necessary environment variables
docker run \
    -e AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id) \
    -e AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key) \
    -e AWS_DEFAULT_REGION=$(aws configure get region) \
    -e AWS_REGION_NAME="$AWS_REGION_NAME" \
    -e RUN_ENV="AWS" \
    -e S3_BUCKET_NAME="$S3_BUCKET_NAME" \
    -e DYNAMODB_TABLE_NAME="$DYNAMODB_TABLE_NAME" \
    -e JOB_ID="$job_id" \
    soso-scheduler:$version
