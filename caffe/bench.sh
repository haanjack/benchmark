#!/bin/bash
COMMAND="bash ./models/run_model.sh"

${COMMAND} inception_v1 1 64 8 real /imagenet/ilsvrc12_train_lmdb /imagenet/ilsvrc12_val_lmdb /imagenet/imagenet_mean.binaryproto
${COMMAND} resnet_101 1 32 8 real /imagenet/ilsvrc12_train_lmdb /imagenet/ilsvrc12_val_lmdb /imagenet/imagenet_mean.binaryproto
