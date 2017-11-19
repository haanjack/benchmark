#!/bin/bash
# NVIDIA Copyright

TASK_INDEX=$1
JOB_NAME="ps"
PORT=50000

if [ -z "$1" ]; then
  echo "Usage: run-ps.sh [task-index]"
  exit
fi

# Create the output and temporary directories.
BENCHMARK_SCRIPT_DIR="$(pwd)"
VERSION=17.11

nvidia-docker run -d -ti --name tensorflow-${JOB_NAME}-${TASK_INDEX} \
    -u $(id -u):$(id -g) \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -p ${PORT}:${PORT} \
    -v ${BENCHMARK_SCRIPT_DIR}:/workspace \
    nvcr.io/nvidia/tensorflow:$VERSION bash bench.sh ${JOB_NAME} ${TASK_INDEX}



