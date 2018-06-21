VERSION=${1}

docker build -t tensorflow:devel-gpu-${VERSION} -f ./dockerfiles/Dockerfile.devel-gpu-${VERSION} .
