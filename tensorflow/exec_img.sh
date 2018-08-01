#!/bin/bash
# Tensorflow benchmark
# Author: Jack Han <jahan@nvidia.com>
#
# Compatible with Tensorflow r1.9
# This script works with provided dockerfile built image since it uses cloned benchmark code
#

dataset_dir=/datasets/imagenet/imagenet_resized352
dataset_dir=/raid/datasets/imagenet/tfrecord
output_dir="output"

docker_image=hanjack/tensorflow
docker_version=devel-gpu-1.9
nv_docker_image=nvcr.io/nvidia/tensorflow
nv_docker_version=18.06-py3

num_epochs=0
num_samples=1281167 # ILSVRC2012

function exec()
{
    image=${1}
    tag=${2}
    bmt_src=${3}
    model=${4}
    batch_size=${5}
    num_gpu=${6}
    use_fp16=${7}
    layers=${8}

    if [[ ${num_epochs} -gt 0 ]]; then
        num_batches=$((${num_epochs} * ${num_samples} / ${batch_size} / ${num_gpu}))
    else
        num_batches=500
    fi

    fp16_nv_option=
    if [[ ${use_fp16} -eq 0 ]]; then
      use_fp16="False"
      precision=fp32
    elif [[ ${use_fp16} -eq 1 ]]; then
      use_fp16="True"
      precision=fp16
      fp16_nv_option="--fp16"
    fi

    bmt_script=
    case ${bmt_src} in
      "tf")
        bmt_script="python /workspace/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
              --model=${model} --batch_size=${batch_size} --num_gpus=${num_gpu} --num_batches=${num_batches} \
              --data_name=imagenet --data_dir=/imagenet \
              --variable_update=replicated --all_reduce_spec=nccl --use_fp16=${use_fp16}"
      ;;
      "nv")
        # bmt_script="python /workspace/nvidia-examples/cnn/nvcnn.py \
        #        --model=${model} --batch_size=${batch_size} --num_gpus=${num_gpu} --num_batches=${num_batches} \
        #        --data_dir=/imagenet ${fp16_nv_option}"
	bmt_script="mpiexec --allow-run-as-root -np ${num_gpu} \
	    python /workspace/nvidia-examples/cnn/${model}.py --layers=${layers} \
	    --data_dir=/imagenet --batch_size=${batch_size} --num_iter=${num_batches} --iter_unit batch --precision=${precision}"
      ;;
    esac

    log_file=${output_dir}/log_${model}_${precision}_b${batch_size}_g${num_gpu}.txt
    echo $bmt_script

    # benchmark code
    start_time="$(date -u +%s)"
    time nvidia-docker run --rm -ti --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
        -u $(id -u):$(id -g) \
        -v ${dataset_dir}:/imagenet \
        ${image}:${tag} \
            ${bmt_script} |& tee ${log_file}
    end_time="$(date -u +%s)"

    elapsed="$(($end_time - $start_time))"
    echo "Total of $elapsed secondes elapsed" |& tee -a ${log_file}

}

# mkdir for tensorflow output saving
if [ ! -d ${output_dir} ]; then
    mkdir ${output_dir}
fi

# Benchmark example
# exec {docker image} {tag} {model-name} {batch_size} {num_gpu} {fp16 (0; false, 1; true)} {layers}
#exec ${nv_docker_image} ${nv_docker_version} nv resnet 64 1 1 50
#exec ${nv_docker_image} ${nv_docker_version} nv resnet 64 2 1 50
#exec ${nv_docker_image} ${nv_docker_version} nv resnet 64 4 1 50
#exec ${nv_docker_image} ${nv_docker_version} nv resnet 64 8 1 50

exec ${docker_image} ${docker_version} tf resnet50 64 1 1
exec ${docker_image} ${docker_version} tf resnet50 64 2 1
exec ${docker_image} ${docker_version} tf resnet50 64 4 1
exec ${docker_image} ${docker_version} tf resnet50 64 8 1
