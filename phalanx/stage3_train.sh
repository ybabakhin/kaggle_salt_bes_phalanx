#!/usr/bin/env bash

# pretrain stage3 model with pseudolabels
python3 train_pseudo.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 64 \
--epoch 150 \
--snapshot 3 \
--cuda True \
--save_weight weights/ \
--max_lr 0.01 \
--min_lr 0.001 \
--momentum 0.9 \
--weight_decay 1e-4 \
--pseudo_path /workdir/data/pseudolabels_v2/

# train stage3 model with train data
python3 train_cv.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 64 \
--epoch 200 \
--snapshot 4 \
--cuda True \
--save_weight /workdir/phalanx/weights/ \
--max_lr 0.01 \
--min_lr 0.001 \
--momentum 0.9 \
--weight_decay 1e-4 \
