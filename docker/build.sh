#!/bin/bash
cd $(dirname $0)

source params.sh

docker build -t "${IMAGENAME}" .