#!/bin/bash
# NVIDIA Copyright

DATASET_DIR="$1"
JOB_NAME="$2"
TASK_INDEX=$3

if [ -z "$1" ]; then
  echo "Usage: docker-run.sh [dataset]"
  exit
fi

# Create the output and temporary directories.
BENCHMARK_SCRIPT_DIR="$(pwd)"
VERSION=17.10

nvidia-docker run --rm -ti --name tensorflow-${JOB_NAME} \
    -u $(id -u):$(id -g) \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 2222:2222 \
    -v ${DATASET_DIR}:/imagenet \
    -v ${BENCHMARK_SCRIPT_DIR}:/workspace \
    nvcr.io/nvidia/tensorflow:$VERSION bash bench.sh ${JOB_NAME} ${TASK_INDEX}



