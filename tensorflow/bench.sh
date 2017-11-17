#/bin/bash

DATA_DIR=/imagenet
LOG_DIR=output
TIMESTAMP=$(date +%m%d%H%M)
NUM_EPOCHS=100
NUM_GPU=8
VARIABLE_UPDATE=distributed_replicated
JOB_NAME=$1
TASK_INDEX=$2

mkdir -p ${LOG_DIR}

train() {
    MODEL=$1
    BATCH_SIZE=$2
    JOB_NAME=$3
    TASK_INDEX=$4
    TIMESTAMP=$(date +%m%d%H%M)
    LOG_FILE="${LOG_DIR}/output_${MODEL}_e${NUM_EPOCHS}_b${BATCH_SIZE}.${TIMESTAMP}.log"
    python tf_cnn_benchmarks/tf_cnn_benchmarks.py \
        --job_name=${JOB_NAME} \
        --ps_hosts="192.168.200.212:2222" \
        --worker_hosts="192.168.200.212:2222" \
        --task_index=${TASK_INDEX} \
        --model=${MODEL} --batch_size=${BATCH_SIZE} --num_baches=${NUM_EPOCHS} --num_gpus=${NUM_GPU} \
        --data_name=imagenet --data_dir=${DATA_DIR} --variable_update=${VARIABLE_UPDATE} \
        2>&1 | tee ${LOG_FILE}
	
}

train googlenet 64 ${JOB_NAME} ${TASK_INDEX}
#train resnet101 32

