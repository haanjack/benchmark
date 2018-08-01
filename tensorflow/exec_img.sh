#!/bin/bash
# Tensorflow benchmark
# Author: Jack Han <jahan@nvidia.com>
#
# Compatible with Tensorflow r1.5/r1.8/r1.9
# This script works with provided dockerfile built image since it uses cloned benchmark code
#

DATASET_DIR=/raid/datasets/imagenet/tfrecord
OUTPUT_DIR="output"
DOCKER_IMAGE=hanjack/tensorflow:devel-gpu-1.9
NUM_ITER=500
ITER_UNIT="batch"

usage() {
    echo "Usage::"
    echo "--data_dir: dataset path"
    echo "--docker_image: docker image"
    echo "--num_iter: number of iteration"
    echo "--output_dir: log output path (default:output)"
    echo "--iter_unit: iteration unit (batch|epoch), (default=batch)"
    echo ""
    echo "Example::"
    echo "./exec_img.sh --data_dir=/raid/datasets/imagenet/tfrecord --docker_image=hanjack/tensorflow:devel-gpu-1.9 --num_iter=500"
    exit 1
}

if [ -z "$@" ]; then
    echo "ASD"
    usage
fi

for i in "$@"; do
    case "$i" in
	--data_dir=*)
	DATASET_DIR="${i#*=}"
	;;
        --docker_image=*)
	DOCKER_IMAGE="${i#*=}"
	;;
        -i=*|--num_iter=*)
	NUM_ITER="${i#*=}"
	;;
	-o=*|--output_dir=*)
	OUTPUT_DIR="${i#*=}"
	;;
	-u|--iter_unit=*)
	ITER_UNIT="${i#*=}"
	;;
        --help)
        usage
        ;;
        *)
	# unknown option
        usage
	;;
    esac
    shift 1
done

echo "dataset_dir: ${DATASET_DIR}"
echo "docker image: ${DOCKER_IMAGE}"
echo "num_iter: ${NUM_ITER}"
echo "output dir: ${OUTPUT_DIR}"
echo "iter unit: ${ITER_UNIT}"

num_epochs=0
num_samples=1281167 # ILSVRC2012

if [ ${ITER_UNIT} -eq "epoch" ]; then
    num_epochs=${num_iter}
fi

function exec()
{
    image=${1}
    bmt_src=${2}
    model=${3}
    batch_size=${4}
    num_gpu=${5}
    use_fp16=${6}
    layers=${7}

    if [[ ${num_epochs} -gt 0 ]]; then
        num_batches=$((${num_epochs} * ${num_samples} / ${batch_size} / ${num_gpu}))
    else
        num_batches=${NUM_ITER}
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

    echo ${bmt_script}
    log_file=${OUTPUT_DIR}/log_${model}_${precision}_b${batch_size}_g${num_gpu}.txt

    # benchmark code
    start_time="$(date -u +%s)"
    time nvidia-docker run --rm -ti --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
        -u $(id -u):$(id -g) \
        -v ${DATASET_DIR}:/imagenet \
        ${image} \
            ${bmt_script} |& tee ${log_file}
    end_time="$(date -u +%s)"

    elapsed="$(($end_time - $start_time))"
    echo "Total of $elapsed secondes elapsed" |& tee -a ${log_file}

}

# mkdir for tensorflow output saving
if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir ${OUTPUT_DIR}
fi

# Benchmark example
# exec {docker image} {tag} {model-name} {batch_size} {num_gpu} {fp16 (0; false, 1; true)} {layers}
#exec ${nv_docker_image} nv resnet 64 1 1 50
#exec ${nv_docker_image} nv resnet 64 2 1 50
#exec ${nv_docker_image} nv resnet 64 4 1 50
#exec ${nv_docker_image} nv resnet 64 8 1 50

exec ${DOCKER_IMAGE} tf resnet50 64 1 1
exec ${DOCKER_IMAGE} tf resnet50 64 2 1
exec ${DOCKER_IMAGE} tf resnet50 64 4 1
exec ${DOCKER_IMAGE} tf resnet50 64 8 1
