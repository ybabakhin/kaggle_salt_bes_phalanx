#!/bin/bash
cd $(dirname $0)

NAME="bes-phalanx-tgs-salt"
IMAGENAME="${NAME}"
CONTNAME="--name=${NAME}"
NET="--net=host"
IPC="--ipc=host"
TEST_DATA="$(python3 -c "import sys, json; print(json.load(open('../SETTINGS.json'))['TEST_IMAGES_DATA'])")"

VOLUMES="-v $(pwd)/..:/workdir -v ${TEST_DATA}:/test_data"