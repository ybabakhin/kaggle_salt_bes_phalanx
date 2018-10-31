
python3 predict_test.py \
--epochs 125 \
--pretrain_weights weights/unet_resnext_50_stage_1_1/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.00005 \
--input_size 224 \
--resize_size 192 \
--batch_size 24 \
--loss_function lovasz \
--callback reduce_lr \
--early_stop_patience 15 \
--reduce_lr_factor 0.5 \
--reduce_lr_patience 7 \
--reduce_lr_min 0.00000625 \
--network unet_resnext_50 \
--alias _lovasz_stage_1_2 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder fold_{}

python3 predict_test.py \
--epochs 40 \
--n_snapshots 1 \
--pretrain_weights weights/unet_resnext_50_lovasz_stage_1_2/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.0001 \
--input_size 224 \
--resize_size 192 \
--batch_size 24 \
--loss_function lovasz \
--callback snapshot \
--network unet_resnext_50 \
--alias _lovasz_stage_1_3 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder fold_{}

python3 predict_test.py \
--epochs 40 \
--n_snapshots 1 \
--pretrain_weights weights/unet_resnext_50_lovasz_stage_1_3/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.0001 \
--input_size 224 \
--resize_size 192 \
--batch_size 24 \
--loss_function lovasz \
--callback snapshot \
--network unet_resnext_50 \
--alias _lovasz_stage_1_4 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder fold_{}

python3 predict_test.py \
--epochs 40 \
--n_snapshots 1 \
--pretrain_weights weights/unet_resnext_50_lovasz_stage_1_4/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.0001 \
--input_size 224 \
--resize_size 192 \
--batch_size 24 \
--loss_function lovasz \
--callback snapshot \
--network unet_resnext_50 \
--alias _lovasz_stage_1_5 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder fold_{}
