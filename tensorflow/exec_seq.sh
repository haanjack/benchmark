#!/bin/bash
# Tensorflow benchmark
# Author: Jack Han <jahan@nvidia.com>
#
# Compatible with Tensorflow @NGC
# This script works with provided dockerfile built image since it uses cloned benchmark code
#

output_dir="result"
summary_freq=100
dataset_dir="/raid/datasets/wmt16_en_dt"
benchmark_flag=1

function exec()
{
    image=${1}
    tag=${2}
    num_gpu=${3}
    batch_size=${4}
    embed_size=${5}

    log_file=${output_dir}/log_mnt_b${batch_size}_e${embed_size}_g${num_gpu}.txt

    nvidia-docker run --rm -ti --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
        --name tensorflow \
        -u $(id -u):$(id -g) \
        -e ${HOME}:${HOME} -e HOME=${HOME} \
        -v ${dataset_dir}:/dataset \
        ${image}:${tag} \
           /workspace/OpenSeq2Seq/try_gnmt_en2de.sh /dataset \
              ${num_gpu} ${batch_size} ${embed_size} ${summary_freq} /dataset ${benchmark_flag} \
              |& tee ${log_file}
}

if [ ! -d ${output_dir} ]; then
    mkdir ${output_dir}
fi

# Benchmark example
# exec {docker image} {tag} {model-name} {num_gpu} {batch_size} {embed_size}
exec jahan/tensorflow 18.02-py2 4 256 1024
