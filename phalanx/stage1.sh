#train stage1 model with train data
python3 train_cv.py \
--model res34v4 \
--fine_size 202 \
--pad_left 27 \
--pad_right 27 \
--batch_size 18 \
--epoch 300 \
--snapshot 6 \
--cuda True \
--save_weight weights/stage1/ \
--max_lr 0.012 \
--min_lr 0.001 \
--momentum 0.9 \
--weight_decay 1e-4 \


#prediction with stage1 model
python3 precisioncv.py \
--model res34v4 \
--fine_size 202 \
--pad_left 27 \
--pad_right 27 \
--batch_size 18 \
--cuda True \
--fold 0 \
--save_weight weights/stage1/ \
--start_snap 1 \
--end_snap 5 \


python3 precisioncv.py \
--model res34v4 \
--fine_size 202 \
--pad_left 27 \
--pad_right 27 \
--batch_size 18 \
--cuda True \
--fold 1
--save_weight weights/stage1/ \
--start_snap 1 \
--end_snap 5 \


python3 precisioncv.py \
--model res34v4 \
--fine_size 202 \
--pad_left 27 \
--pad_right 27 \
--batch_size 18 \
--cuda True \
--fold 2
--save_weight weights/stage1/ \
--start_snap 1 \
--end_snap 5 \


python3 precisioncv.py \
--model res34v4 \
--fine_size 202 \
--pad_left 27 \
--pad_right 27 \
--batch_size 18 \
--cuda True \
--fold 3
--save_weight weights/stage1/ \
--start_snap 1 \
--end_snap 5 \


python3 precisioncv.py \
--model res34v4 \
--fine_size 202 \
--pad_left 27 \
--pad_right 27 \
--batch_size 18 \
--cuda True \
--fold 4
--save_weight weights/stage1/ \
--start_snap 1 \
--end_snap 5 \


#submit ensemble prediction
python3 submit34.py \
--model res34v4
--pred_path predictions/phalanx_stage_1.npy
