#!/usr/bin/env bash

# fold predictions
python3 precisioncv.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 36 \
--cuda True \
--fold 0 \
--save_weight weights/ \
--start_snap 0 \
--end_snap 3 \

python3 precisioncv.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 36 \
--cuda True \
--fold 1 \
--save_weight weights/ \
--start_snap 0 \
--end_snap 3 \

python3 precisioncv.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 36 \
--cuda True \
--fold 2 \
--save_weight weights/ \
--start_snap 0 \
--end_snap 3 \


python3 precisioncv.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 36 \
--cuda True \
--fold 3 \
--save_weight weights/ \
--start_snap 0 \
--end_snap 2 \


python3 precisioncv.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 36 \
--cuda True \
--fold 4 \
--save_weight weights/ \
--start_snap 0 \
--end_snap 2 \

# stage3 model prediction
python3 submit34.py \
--model res34v5 \
--pred_path predictions/phalanx_stage_3.npy
