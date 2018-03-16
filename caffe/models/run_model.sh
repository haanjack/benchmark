#!/usr/bin/env bash

MODEL="$1"
MAXITER="$2"
BATCHSIZE="${3:-32}"
FP16=${4:-0}
NGPUS=${5:-1}
IMAGENET_TRAIN="${6:-/imagenet/ilsvrc12_train_lmdb}"
IMAGENET_VAL="${7:-/imagenet/ilsvrc12_val_lmdb}"
IMAGENET_MEAN="${8:-/imagenet/imagenet_mean.binaryproto}"
LOG_DIR=./output
VERBOSE_RUN=${VERBOSE_RUN:-0}

SAMPLE_SIZE=1281167
MAXGPUS="$(nvidia-smi -L | wc -l)"

if [[ ${NGPUS} -gt ${MAXGPUS} ]]; then
    echo "requested to large number of gpus"
    exit
elif [[ ${NGPUS} -lt 1 ]]; then
    echo "request at least 1 gpu"
    exit
fi

case "$NGPUS" in
        1) GPU_OPT=--gpu=0
        ;;
        2) GPU_OPT=--gpu=0,1
        ;;
        4) GPU_OPT=--gpu=0,1,2,3
        ;;
        8) GPU_OPT=--gpu=0,1,2,3,4,5,6,7
        ;;
esac

MODEL_DIR="./models/${MODEL}"

CAFFE_BIN=${CAFFE_BIN:-"/usr/local/bin/caffe"}

FLAG_FP16=
if [[ ${FP16} -eq 1 ]]; then
    FLAG_FP16="_fp16"
fi

SOLVER="iter${MAXITER}_b${BATCHSIZE}_${NGPUS}gpu"
SOLVER_TMPL_FILE="${MODEL_DIR}/solver_tmpl.prototxt"
SOLVER_FILE="${MODEL_DIR}/solver_${SOLVER}.prototxt"
LOG_FILE="${LOG_DIR}/output_${MODEL}_${SOLVER}${FLAG_FP16}.log"
TRAIN_TMPL_FILE=${MODEL_DIR}/train_val${FLAG_FP16}_tmpl.prototxt
TRAIN_VAL_FILE=${MODEL_DIR}/train_val_b${BATCHSIZE}${FLAG_FP16}.prototxt

# Caffe data layer needs total batch size,
# whereas dummy data layer needs per gpu batch size
if [[ $DATA_KIND == fake ]]; then
    TRAIN_BATCHSIZE=${BATCHSIZE}
else
    TRAIN_BATCHSIZE=$((BATCHSIZE*NGPUS))
fi
VAL_BATCHSIZE=${BATCHSIZE}

cleanup() {
    rm -f ${TRAIN_VAL_FILE} ${SOLVER_FILE}
    rm -f *.solverstate *.caffemodel
}

# Assuming that @ is not used in train_val files
cat ${TRAIN_TMPL_FILE} | sed "s/REPLACE_TRAIN_BATCH_SIZE/${TRAIN_BATCHSIZE}/g" | sed "s/REPLACE_VAL_BATCH_SIZE/${VAL_BATCHSIZE}/g" | \
    sed "s@REPLACE_IMAGENET_TRAIN@${IMAGENET_TRAIN}@g" | sed "s@REPLACE_IMAGENET_VAL@${IMAGENET_VAL}@g" | \
    sed "s@REPLACE_IMAGENET_MEAN@${IMAGENET_MEAN}@g" > ${TRAIN_VAL_FILE}

cat ${SOLVER_TMPL_FILE} | sed "s/REPLACE_MAX_ITER/${MAXITER}/g" | sed "s^REPLACEMODELDIR^${MODEL_DIR}^g" | \
    sed "s@REPLACE_TRAIN_VAL_FILE@${TRAIN_VAL_FILE}@g" > ${SOLVER_FILE}

mkdir -p ${LOG_DIR}
${CAFFE_BIN}  train --solver=${SOLVER_FILE} ${GPU_OPT} 2>&1 | tee ${LOG_FILE}

cleanup
