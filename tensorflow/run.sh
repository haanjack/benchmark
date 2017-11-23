#!/bin/bash
# NVIDIA Copyright

DATASET_DIR="$1"
JOB_NAME="single"

if [ -z "$1" ]; then
  echo "Usage: docker-run.sh [dataset]"
  exit
fi

# Create the output and temporary directories.
BENCHMARK_SCRIPT_DIR="$(pwd)"
VERSION=17.11

NV_GPU=0,1 nvidia-docker run --rm -ti --name tensorflow-${JOB_NAME} \
    -u $(id -u):$(id -g) \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    --net=host --privileged \
    -v ${DATASET_DIR}:/imagenet \
    -v ${BENCHMARK_SCRIPT_DIR}:/workspace \
    nvcr.io/nvidia/tensorflow:$VERSION bash bench.1node.sh ${JOB_NAME} 



