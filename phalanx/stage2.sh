# train stage2 model with train data and pseudolabels
python3 train_cv.py \
--model res34v3 \
--fine_size 202 \
--pad_left 27 \
--pad_right 27 \
--batch_size 18 \
--epoch 200 \
--snapshot 4 \
--cuda True \
--save_weight weights/ \
--max_lr 0.012 \
--min_lr 0.001 \
--momentum 0.9 \
--weight_decay 1e-4 \
--is_pseudo True \


# fold predictions
python3 precisioncv.py \
--model res34v3 \
--fine_size 202 \
--pad_left 27 \
--pad_right 27 \
--batch_size 18 \
--cuda True \
--fold 0 \
--save_weight weights/ \
--start_snap 0 \
--end_snap 3 \


python3 precisioncv.py \
--model res34v3 \
--fine_size 202 \
--pad_left 27 \
--pad_right 27 \
--batch_size 18 \
--cuda True \
--fold 1 \
--save_weight weights/ \
--start_snap 0 \
--end_snap 3 \


python3 precisioncv.py \
--model res34v3 \
--fine_size 202 \
--pad_left 27 \
--pad_right 27 \
--batch_size 18 \
--cuda True \
--fold 2 \
--save_weight weights/ \
--start_snap 0 \
--end_snap 3 \


python3 precisioncv.py \
--model res34v3 \
--fine_size 202 \
--pad_left 27 \
--pad_right 27 \
--batch_size 18 \
--cuda True \
--fold 3 \
--save_weight weights/ \
--start_snap 0 \
--end_snap 3 \


python3 precisioncv.py \
--model res34v3 \
--fine_size 202 \
--pad_left 27 \
--pad_right 27 \
--batch_size 18 \
--cuda True \
--fold 4 \
--save_weight weights/ \
--start_snap 0 \
--end_snap 3 \


# stage2 model prediction
python3 submit34.py \
--model res34v3 \
--pred_path predictions/phalanx_stage_2.npy
