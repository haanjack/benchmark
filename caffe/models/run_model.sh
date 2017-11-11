#!/usr/bin/env bash

MODEL="$1"
EPOCHS="$2"
BATCHSIZE="$3"
NGPUS=${4:-0}
DATA_KIND="$5"
IMAGENET_TRAIN="$6"
IMAGENET_VAL="$7"
IMAGENET_MEAN="$8"
LOG_DIR=./output
VERBOSE_RUN=${VERBOSE_RUN:-0}

SAMPLE_SIZE=1281167
MAXGPUS="$(nvidia-smi -L | wc -l)"

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

MAXITER=$((${SAMPLE_SIZE} * ${EPOCHS} / ${BATCHSIZE} / ${NGPUS}))
MAXITER=1000

MODEL_DIR="./models/${MODEL}"

CAFFE_BIN=${CAFFE_BIN:-"/usr/local/bin/caffe"}

SOLVER="iter${MAXITER}_b${BATCHSIZE}_${NGPUS}gpu"
SOLVER_TMPL_FILE="${MODEL_DIR}/solver_tmpl.prototxt"
SOLVER_FILE="${MODEL_DIR}/solver_${SOLVER}.prototxt"
TIMESTAMP=$(date +%m%d%H%M)
LOG_FILE="${LOG_DIR}/output_${MODEL}_${SOLVER}.${TIMESTAMP}.log"

TRAIN_TMPL_FILE=${MODEL_DIR}/train_val_tmpl_${DATA_KIND}_data.prototxt
TRAIN_VAL_FILE=${MODEL_DIR}/train_val_${DATA_KIND}_data_b${BATCHSIZE}.prototxt

# Caffe data layer needs total batch size,
# whereas dummy data layer needs per gpu batch size
if [[ $DATA_KIND == fake ]]; then
    TRAIN_BATCHSIZE=${BATCHSIZE}
else
    TRAIN_BATCHSIZE=$((BATCHSIZE*NGPUS))
fi

cleanup() {
    rm -f ${TRAIN_VAL_FILE} ${SOLVER_FILE}
    rm -f *.solverstate *.caffemodel
}

# Assuming that @ is not used in train_val files
cat ${TRAIN_TMPL_FILE} | sed "s/REPLACEBATCHSIZE/${TRAIN_BATCHSIZE}/g" | sed "s@REPLACEIMAGENETMEAN@${IMAGENET_MEAN}@g" | sed "s@REPLACEIMAGENETTRAIN@${IMAGENET_TRAIN}@g" | sed "s@REPLACEIMAGENETVAL@${IMAGENET_VAL}@g" | sed "s@transform_param {@transform_param {${GPU_TRANSFORM}@g" > ${TRAIN_VAL_FILE}

cat ${SOLVER_TMPL_FILE} | sed "s/REPLACEBATCHSIZE/${BATCHSIZE}/g" | sed "s/REPLACEMAXITER/${MAXITER}/g" | sed "s^REPLACEMODELDIR^${MODEL_DIR}^g" | sed "s^REPLACEDATAKIND^${DATA_KIND}_data^g" > ${SOLVER_FILE}

mkdir -p ${LOG_DIR}
${CAFFE_BIN}  train --solver=${SOLVER_FILE} ${GPU_OPT} 2>&1 | tee ${LOG_FILE}

cleanup
