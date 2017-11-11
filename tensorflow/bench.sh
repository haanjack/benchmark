#!/bin/bash

LOG_DIR=./output
DATA_DIR=/imagenet
NUM_EPOCHS=1

mkdir -p ${LOG_DIR}

MODEL=googlenet
BATCH_SIZE=64
TIMESTAMP=$(date +%m%d%H%M)
LOG_FILE="${LOG_DIR}/output_${MODEL}_e${NUM_EPOCHS}_b${BATCH_SIZE}.tfevent.${TIMESTAMP}.log"
python nvcnn.py --model=${MODEL} --batch_size=${BATCH_SIZE} --num_epochs=${NUM_EPOCHS} \
  --data_dir=${DATA_DIR} --num_gpus=8 2>&1 | tee ${LOG_FILE}

MODEL=resnet101
BATCH_SIZE=32
TIMESTAMP=$(date +%m%d%H%M)
LOG_FILE="${LOG_DIR}/output_${MODEL}_e${NUM_EPOCHS}_b${BATCH_SIZE}.tfevent.${TIMESTAMP}.log"
python nvcnn.py --model=${MODEL} --batch_size=${BATCH_SIZE} --num_epochs=${NUM_EPOCHS} \
  --data_dir=${DATA_DIR} --num_gpus=8 2>&1 | tee ${LOG_FILE}
