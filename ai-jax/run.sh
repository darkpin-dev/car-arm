#!/bin/bash

CONTAINER_NAME="jax-rocm-dev"
IMAGE_NAME="rocm/jax:rocm5.7.0-jax0.4.26-py3.11.0"

if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then

    if [ "$(docker ps -q -f name=^/${CONTAINER_NAME}$)" ]; then
        echo "'${CONTAINER_NAME}' container is running"
    else
        echo "run container"
        docker start ${CONTAINER_NAME}
    fi

else
    echo "create container"
    docker run -d \
      --name ${CONTAINER_NAME} \
        --network=host \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --ipc=host \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
        -v $(pwd):/workspace \
        -w /workspace \
        -e HSA_OVERRIDE_GFX_VERSION=9.0.6 \
        -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
      ${IMAGE_NAME} \
        tail -f /dev/null
fi

xhost +local:docker