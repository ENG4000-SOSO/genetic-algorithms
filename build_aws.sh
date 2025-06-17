#!/bin/bash

# -----------------------------------------------------------------------------
# This script builds and pushes a Docker image of the SOSO scheduler service to
# an AWS Elastic Container Registry (ECR) repository.
#
# Usage:
#   ./push_scheduler.sh <version> [--skip-build]
#
# Arguments:
#
#   <version>      - Required. The version tag to assign to the Docker image.
#
#   --skip-build   - Optional. If provided, skips the Docker build step and
#                      only tags and pushes an existing local image.
#
#   Behavior:
#     1. Logs in to the specified AWS ECR registry.
#     2. Builds the Docker image using Dockerfile.aws (unless --skip-build is
#          set).
#     3. Tags the image with the provided version.
#     4. Pushes the tagged image to the AWS ECR repository.
#
# Requirements:
#   - AWS CLI configured with credentials and permissions to push to ECR.
#   - Docker installed and running.
# -----------------------------------------------------------------------------

# Check if a version argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <version> [--skip-build]"
    exit 1
fi

# Extract version number and shift positional arguments
version=$1
shift

# Authenticate Docker to AWS ECR
aws ecr get-login-password | docker login --username AWS --password-stdin 607869540801.dkr.ecr.us-east-1.amazonaws.com/soso-ecr-1

# Initialize skip_build flag
skip_build=false

# Process optional arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build)
      skip_build=true
      echo "Skipping the build step."
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 <version> [--skip-build]"
      exit 1
      ;;
  esac
  shift
done

# Build Docker image (unless --skip-build is specified)
if [ "$skip_build" = false ]; then
    docker build -f Dockerfile.aws -t soso-scheduler:$version .
fi

# Tag the built image with the full ECR registry path and version
docker tag soso-scheduler:$version 607869540801.dkr.ecr.us-east-1.amazonaws.com/soso-ecr-1:$version

# Push the tagged image to AWS ECR
docker push 607869540801.dkr.ecr.us-east-1.amazonaws.com/soso-ecr-1:$version
