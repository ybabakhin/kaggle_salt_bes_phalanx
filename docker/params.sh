#!/bin/bash
cd $(dirname $0)

NAME="bes-phalanx-tgs-salt"
IMAGENAME="${NAME}"
CONTNAME="--name=${NAME}"
NET="--net=host"
IPC="--ipc=host"
TEST_DATA="$(python3 -c "import sys, json; print(json.load(open('../SETTINGS.json'))['NEW_TEST_IMAGES_DATA'])")"
if [ "$TEST_DATA" == "/workdir/data/test/images/" ]; then TEST_DATA="$(pwd)/../data/test/images/"; fi

VOLUMES="-v $(pwd)/..:/workdir -v ${TEST_DATA}:/test_data"