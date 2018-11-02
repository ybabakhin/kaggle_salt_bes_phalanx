
# Train Stage 1 Models
pushd phalanx
./stage1.sh
popd

pushd bes
./stage1.sh

# Generate Pseudolabels
python3 ensemble.py \
--stage 1

popd

# Train Stage 2 Models
pushd phalanx

# Generate pseudolabels for folds
python3 make_pseudo.py
./stage2.sh
popd

pushd bes
./stage2_train.sh
./stage2_predict.sh

# Generate Pseudolabels
python3 ensemble.py \
--stage 2

popd

# Train Stage 3 Models
pushd phalanx
./stage3_train.sh
popd
