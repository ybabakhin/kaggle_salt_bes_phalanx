python3 train.py \
--epochs 125 \
--fold 0,1,2,3,4 \
--learning_rate 0.0001 \
--input_size 224 \
--resize_size 192 \
--batch_size 24 \
--loss_function bce_dice \
--callback reduce_lr \
--early_stop_patience 20 \
--reduce_lr_factor 0.25 \
--reduce_lr_patience 10 \
--reduce_lr_min 0.00000625 \
--network unet_resnext_50 \
--alias _192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual


python3 train.py \
--epochs 125 \
--pretrain_weights weights/unet_resnext_50_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual/fold_{}.hdf5 \
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
--network unet_resnext_50_lovasz \
--alias _192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005


# Snapshots
python3 train.py \
--epochs 40 \
--n_snapshots 1 \
--pretrain_weights weights/unet_resnext_50_lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.0001 \
--input_size 224 \
--resize_size 192 \
--batch_size 24 \
--loss_function lovasz \
--callback snapshot \
--network unet_resnext_50_lovasz \
--alias _192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v1

python3 train.py \
--epochs 40 \
--n_snapshots 1 \
--pretrain_weights weights/unet_resnext_50_lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v1/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.0001 \
--input_size 224 \
--resize_size 192 \
--batch_size 24 \
--loss_function lovasz \
--callback snapshot \
--network unet_resnext_50_lovasz \
--alias _192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v2

python3 train.py \
--epochs 40 \
--n_snapshots 1 \
--pretrain_weights weights/unet_resnext_50_lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v2/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.0001 \
--input_size 224 \
--resize_size 192 \
--batch_size 24 \
--loss_function lovasz \
--callback snapshot \
--network unet_resnext_50_lovasz \
--alias _192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v3