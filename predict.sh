#!/usr/bin/env bash

pushd phalanx
./stage3_predict.sh
popd


pushd bes
./stage2_predict.sh

# Generate Final Predictions
python3 ensemble.py \
--stage 3 \
--postprocessing 1

popd
