#pretrain stage3 model with pseudo labels
python train_pseudo.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 64 \
--epoch 150 \
--snapshot 3 \
--cuda True \
-save_weight ../weights/stage3/ \
--max_lr 0.01 \
--min_lr 0.001 \
--momentum 0.9 \
--weight_decay 1e-4 \
--pseudo_path ../input/pseudolabels_v2/

#train stage3 model with train data
python train_cv.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 64 \
--epoch 200 \
--snapshot 4 \
--cuda True \
--save_weight ../weights/stage3/ \
--max_lr 0.01 \
--min_lr 0.001 \
--momentum 0.9 \
--weight_decay 1e-4 \

#prediction with stage3 model
python precisioncv.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 32 \
--cuda True \
--fold 0 \
--save_weight ../weights/stage3/ \
--start_snap 0 \
--end_snap 3 \

python precisioncv.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 32 \
--cuda True \
--fold 1 \
--save_weight ../weights/stage3/ \
--start_snap 0 \
--end_snap 3 \

python precisioncv.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 32 \
--cuda True \
--fold 2 \
--save_weight ../weights/stage3/ \
--start_snap 0 \
--end_snap 3 \


python precisioncv.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 32 \
--cuda True \
--fold 3 \
--save_weight ../weights/stage3/ \
--start_snap 0 \
--end_snap 2 \


python precisioncv.py \
--model res34v5 \
--fine_size 101 \
--pad_left 13 \
--pad_right 14 \
--batch_size 32 \
--cuda True \
--fold 4 \
--save_weight ../weights/stage3/ \
--start_snap 0 \
--end_snap 3 \

#submit ensemble prediction
python submit.py \
--model res34v5