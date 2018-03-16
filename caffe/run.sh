#!/bin/bash

docker_image="nvcr.io/nvidia/caffe:18.02-py2"
workspace="/opt/benchmark"

docker_run="nvidia-docker run --rm -ti -u $(id -u):$(id -g)"
docker_dataset="-v /raid/datasets/imagenet/lmdb:/imagenet"
docker_workspace="-w ${workspace} -v $(pwd):${workspace}"
docker_command="/${workspace}/bench.sh"

exec ${docker_run} ${docker_dataset} ${docker_workspace} ${docker_image} ${docker_command}
