#!/bin/bash
# NVIDIA Copyright

if [ -z "$1" ]; then
  echo "Usage: docker-run.sh [dataset]"
  exit
fi

# Create the output and temporary directories.
DATASET_DIR="$1"
COMMAND="$2"

nvidia-docker run --rm -ti --name tensorflow \
    -u $(id -u):$(id -g) \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${DATASET_DIR}:/data \
    tf-bench ${COMMAND}


