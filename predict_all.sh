python3 predict_test.py \
--epochs 40 \
--n_snapshots 1 \
--pretrain_weights /home/RnD/babakhin.y/salt/unet_resnext_50_lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005/fold_{}.hdf5 \
--fold 1,2,3,4 \
--learning_rate 0.0001 \
--input_size 224 \
--resize_size 192 \
--batch_size 24 \
--loss_function lovasz \
--callback snapshot \
--freeze_encoder 0 \
--augmentation_name valid_plus \
--augmentation_prob 1.0 \
--network unet_resnext_50 \
--alias _lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v1 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder fold_{}

python3 predict_test.py \
--epochs 40 \
--n_snapshots 1 \
--pretrain_weights /home/RnD/babakhin.y/salt/unet_resnext_50_lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v1/fold_{}.hdf5 \
--fold 1,2,3,4 \
--learning_rate 0.0001 \
--input_size 224 \
--resize_size 192 \
--batch_size 24 \
--loss_function lovasz \
--callback snapshot \
--freeze_encoder 0 \
--augmentation_name valid_plus \
--augmentation_prob 1.0 \
--network unet_resnext_50 \
--alias _lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v2 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder fold_{}


python3 predict_test.py \
--epochs 40 \
--n_snapshots 1 \
--pretrain_weights /home/RnD/babakhin.y/salt/unet_resnext_50_lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v2/fold_{}.hdf5 \
--fold 1,2,3,4 \
--learning_rate 0.0001 \
--input_size 224 \
--resize_size 192 \
--batch_size 24 \
--loss_function lovasz \
--callback snapshot \
--freeze_encoder 0 \
--augmentation_name valid_plus \
--augmentation_prob 1.0 \
--network unet_resnext_50 \
--alias _lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v3 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder fold_{}
