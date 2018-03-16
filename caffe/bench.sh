#!/bin/bash

train_command="./models/run_model.sh"

# {command} {model} {num_iter} {batch_size} {fp16} {num_gpu}
${train_command} resnet50 500 32 0 1
${train_command} resnet50 500 32 1 1 
