#!/bin/bash

source params.sh

nvidia-docker run --rm -it \
    ${NET} \
    ${IPC} \
    ${VOLUMES} \
    ${CONTNAME} \
    ${IMAGENAME}  \
bash