#!/bin/bash
# NVIDIA Copyright

if [ -z "$2" ]; then
  echo "Usage: docker-run.sh [framework] [dataset]"
  exit
fi

# Create the output and temporary directories.
FRAMEWORK="$1"
DATASET_DIR="$2"
BENCHMARK_SCRIPT_DIR="$(pwd)"
VERSION=17.10

nvidia-docker run --rm -ti --name $FRAMEWORK \
    -u $(id -u):$(id -g) \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${DATASET_DIR}:/data \
    -v ${BENCHMARK_SCRIPT_DIR}:/script \
    nvcr.io/nvidia/$FRAMEWORK:$VERSION


