VERSION=${1}

docker build -t hanjack/tensorflow:devel-gpu-${VERSION} -f ./dockerfiles/Dockerfile.devel-gpu --build-arg TF_VERSION=${VERSION} .
