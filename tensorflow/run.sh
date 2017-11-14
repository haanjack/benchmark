#!/bin/bash
# NVIDIA Copyright

if [ -z "$1" ]; then
  echo "Usage: docker-run.sh [dataset]"
  exit
fi

# Create the output and temporary directories.
DATASET_DIR="$1"
BENCHMARK_SCRIPT_DIR="$(pwd)"
VERSION=17.10

nvidia-docker run --rm -ti --name tensorflow \
    -u $(id -u):$(id -g) \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${DATASET_DIR}:/imagenet \
    -v ${BENCHMARK_SCRIPT_DIR}:/workspace \
    nvcr.io/nvidia/tensorflow:$VERSION 



