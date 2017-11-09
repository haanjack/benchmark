python -m\
tf_cnn_benchmarks.tf_cnn_benchmarks --local_parameter_device=gpu --num_gpus=1 \
--batch_size=64 --model=resnet50 --variable_update=replicated \
--data_dir=/data --data_name=imagenet \
--num_epochs=1 \
--task_index=0 \
--ps_host=127.0.0.1:50000 \
--worker_hosts=127.0.0.1:50000
