#!/bin/bash
cd $(dirname $0)

NAME="bes-phalanx-tgs-salt"
IMAGENAME="${NAME}"
CONTNAME="--name=${NAME}"
NET="--net=host"
IPC="--ipc=host"

VOLUMES="-v $(pwd)/..:/workdir"