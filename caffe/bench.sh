#!/bin/bash
COMMAND="bash ./models/run_model.sh"
NUM_EPOCHS=1

${COMMAND} inception_v1 ${NUM_EPOCHS} 64 8 real /imagenet/ilsvrc12_train_lmdb /imagenet/ilsvrc12_val_lmdb /imagenet/imagenet_mean.binaryproto
${COMMAND} inception_v1 ${NUM_EPOCHS} 96 8 real /imagenet/ilsvrc12_train_lmdb /imagenet/ilsvrc12_val_lmdb /imagenet/imagenet_mean.binaryproto
${COMMAND} resnet_101 ${NUM_EPOCHS} 32 8 real /imagenet/ilsvrc12_train_lmdb /imagenet/ilsvrc12_val_lmdb /imagenet/imagenet_mean.binaryproto
