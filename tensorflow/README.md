# TensorFlow Benchmark 

This repository helps people to benchmark their system performance in DL. This code covers tests below;
* ImageNet model based test
* OpenSeq model based test

## System configuration

Before benchmark you need to setup your system.
* NVIDIA GPUs later than pascal. I don't test this code prior to Pascal architecture.
* [NVIDIA GPU driver installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
* [docker installation](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [nvidia-docker installation](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))

## Benchmark preparation
* Getting Dataset(142GB) [Link](https://drive.google.com/open?id=1Die_JtnzJDn9OsBQFzKV4kk_TLijj4AW)
* Pulling test code
  ```bash
  $ git clone https://github.com/haanjack/benchmarks
  ```

## Benchmark
### 1. ImageNet Benchmark

#### Simple start (ResNet50, GPU 1-8)
```
$ ./exec_img.sh --data_dir={PATH to DATASET}
```

#### Test with TF 1.8
```
$ ./exec_img --data_dir={PATH to DATASET} --docker_image=hanjack/tensorflow:devel-gpu-1.8
```

#### Time estimation of 1 epoch 
```
$ ./exec_img --data_dir={PATH to DATASET} --iter_unit=epoch --num_iter=1
```

#### Test your own benchmark scenario
The end of ```exec_img.sh``` is a description of the benchmarks. Please review and modify if you want to have other tests
```
# exec {docker image} {script option} {model-name} {batch_size} {num_gpu} {fp16 (0; false, 1; true)} {layers}
exec ${DOCKER_IMAGE} tf resnet50 64 1 1
exec ${DOCKER_IMAGE} tf resnet50 64 2 1
exec ${DOCKER_IMAGE} tf resnet50 64 4 1
exec ${DOCKER_IMAGE} tf resnet50 64 8 1
```

For example, if you want to test googlenet with FP32,
```
exec ${DOCKER_IMAGE} tf inception_v1 64 1 0
exec ${DOCKER_IMAGE} tf inception_v1 64 2 0
exec ${DOCKER_IMAGE} tf inception_v1 64 4 0
exec ${DOCKER_IMAGE} tf inception_v1 64 8 0
```

2. OpenSeq2Seq Benchmark

To be continue...

## Build your own image

This repository provides docker image build scripts. They also clones benchmark codes corresponding versions.

### Docker image based on TensorFlow Dockerfile
```
$ ./build-tf.sh 1.9 # TensorFlow 1.9
$ ./build-tf.sh 1.8 # TensorFlow 1.8
```
This script regards you will use python3.

### Docker image based on NGC
```
$ ./build-ngc.sh 18.06 py3
$ ./build-ngc.sh 18.06 py2
```

This output image also has NGC benchmark code. So you can try another benchmark from ```exec_img.sh``` with ```nv``` script option.
