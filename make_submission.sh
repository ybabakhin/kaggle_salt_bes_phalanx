#!/usr/bin/env bash

pushd bes

# Generate Final Predictions
python3 ensemble.py \
--stage 3 \
--postprocessing 1 \
--test_predictions_path /workdir/predictions/test_predictions.csv

popd
