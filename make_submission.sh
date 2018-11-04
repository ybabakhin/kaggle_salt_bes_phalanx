
pushd bes

# Generate Final Predictions
python3 ensemble.py \
--stage 3 \
--postprocessing 1

popd
