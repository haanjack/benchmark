#!/bin/bash
# Tensorflow benchmark
# Author: Jack Han <jahan@nvidia.com>
#
# This code helps one who doesn't know how to build docker image
#

base_ver=${1}
python_ver=${2}
tensorflow_ver=${3}
dockerfile=./dockerfiles/Dockerfile.hvd
image_name=benchmark/tensorflow

if [[ ${base_ver} == "" ]]; then
    base_ver="18.06"
fi

if [[ ${python_ver} == "" ]]; then
    python_ver="py3"
fi

docker build \
    -t ${image_name}:${base_ver}-${python_ver} \
    -f ${dockerfile} \
    --build-arg BASE_VERSION=${base_ver} \
    --build-arg PYTHON_VERSION=${python_ver} \
    --build-arg TF_VERSION=${tensorflow_ver} \
    .
