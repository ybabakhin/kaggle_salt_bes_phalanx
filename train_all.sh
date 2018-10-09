# python3 train.py \
# --epochs 125 \
# --fold 0 \
# --learning_rate 0.0001 \
# --input_size 224 \
# --resize_size 192 \
# --batch_size 24 \
# --loss_function bce_dice \
# --callback reduce_lr \
# --freeze_encoder 0 \
# --early_stop_patience 20 \
# --reduce_lr_factor 0.25 \
# --reduce_lr_patience 10 \
# --reduce_lr_min 0.00000625 \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnext_50 \
# --alias _192_224_mosaic_folds_initial
#****************************************************************
# python3 train.py \
# --epochs 125 \
# --fold 0 \
# --learning_rate 0.0001 \
# --input_size 224 \
# --resize_size 192 \
# --batch_size 24 \
# --loss_function bce_dice \
# --callback reduce_lr \
# --freeze_encoder 0 \
# --early_stop_patience 20 \
# --reduce_lr_factor 0.25 \
# --reduce_lr_patience 10 \
# --reduce_lr_min 0.00000625 \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnext_50 \
# --alias _192_224_reduce_lr_smaller_decoder_bce_dice_csse_corrected

# python3 train.py \
# --epochs 80 \
# --n_snapshots 2 \
# --fold 0 \
# --learning_rate 0.0001 \
# --input_size 224 \
# --resize_size 192 \
# --batch_size 24 \
# --loss_function bce_dice \
# --callback snapshot \
# --freeze_encoder 0 \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnext_50 \
# --alias _192_224_reduce_lr_smaller_decoder_bce_dice_csse_snapshot

# python3 train.py \
# --epochs 40 \
# --n_snapshots 1 \
# --pretrain_weights /home/RnD/babakhin.y/salt/unet_resnext_50_lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005/fold_{}.hdf5 \
# --fold 0 \
# --learning_rate 0.0001 \
# --input_size 224 \
# --resize_size 192 \
# --batch_size 24 \
# --loss_function lovasz \
# --callback snapshot \
# --freeze_encoder 0 \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnext_50_lovasz \
# --alias _192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_higher_lr

#**************************************************************

# Snapshots


python3 train.py \
--epochs 100 \
--n_snapshots 2 \
--pretrain_weights /home/RnD/babakhin.y/salt/unet_resnext_50_lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v3/fold_{}.hdf5 \
--fold 0 \
--learning_rate 0.00005 \
--input_size 224 \
--resize_size 192 \
--batch_size 24 \
--loss_function lovasz \
--callback snapshot \
--freeze_encoder 0 \
--augmentation_name horizontal_flip \
--augmentation_prob 1.0 \
--network unet_resnext_50_lovasz \
--alias _192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v3_finetune_hflip


#**************************************************************

# python3 train.py \
# --epochs 150 \
# --fold 0 \
# --learning_rate 0.0001 \
# --input_size 224 \
# --resize_size 192 \
# --batch_size 32 \
# --loss_function bce_dice \
# --callback reduce_lr \
# --freeze_encoder 0 \
# --early_stop_patience 50 \
# --reduce_lr_factor 0.25 \
# --reduce_lr_patience 100 \
# --reduce_lr_min 0.00000625 \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnet_34 \
# --alias _192_224_reduce_lr_smaller_decoder_bce_dice_csse_150_epochs


# python3 train.py \
# --epochs 125 \
# --fold 0 \
# --pretrain_weights /home/RnD/babakhin.y/salt/unet_resnet_34_192_224_reduce_lr_smaller_decoder_bce_dice_hypecolumn_csse/fold_{}.hdf5 \
# --learning_rate 0.00005 \
# --input_size 224 \
# --resize_size 192 \
# --batch_size 16 \
# --loss_function lovasz \
# --callback reduce_lr \
# --freeze_encoder 0 \
# --early_stop_patience 15 \
# --reduce_lr_factor 0.5 \
# --reduce_lr_patience 8 \
# --reduce_lr_min 0.0000125 \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnet_34_lovasz \
# --alias _192_224_reduce_lr_smaller_decoder_bce_dice_hypecolumn_csse

# python3 train.py \
# --epochs 125 \
# --fold 0 \
# --learning_rate 0.0001 \
# --input_size 128 \
# --resize_size 101 \
# --batch_size 64 \
# --loss_function bce_dice \
# --callback reduce_lr \
# --freeze_encoder 0 \
# --early_stop_patience 20 \
# --reduce_lr_factor 0.25 \
# --reduce_lr_patience 10 \
# --reduce_lr_min 0.00000625 \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnet_34 \
# --alias _101_128_reduce_lr_smaller_decoder_bce_dice_corr

# python3 train.py \
# --epochs 50 \
# --n_snapshots 1 \
# --fold 0 \
# --learning_rate 0.0001 \
# --input_size 224 \
# --resize_size 192 \
# --batch_size 32 \
# --loss_function bce_dice \
# --callback snapshot \
# --freeze_encoder 0 \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnet_34 \
# --alias _192_224_v4

# python3 train.py \
# --epochs 40 \
# --pretrain_weights /home/RnD/babakhin.y/salt/unet_resnet_34_lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_150_epochs/fold_{}.hdf5 \
# --n_snapshots 2 \
# --fold 0 \
# --learning_rate 0.00005 \
# --input_size 224 \
# --resize_size 192 \
# --batch_size 32 \
# --loss_function lovasz \
# --callback snapshot \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnet_34_lovasz \
# --alias _192_224_reduce_lr_smaller_decoder_bce_dice_csse_150_epochs_2_snapshots




