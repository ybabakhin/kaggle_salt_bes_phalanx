python3 predict_oof.py \
--epochs 125 \
--pretrain_weights /home/RnD/babakhin.y/salt/unet_resnext_50_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual/fold_{}.hdf5 \
--fold 0 \
--learning_rate 0.00005 \
--input_size 224 \
--resize_size 192 \
--batch_size 24 \
--loss_function lovasz \
--callback reduce_lr \
--freeze_encoder 0 \
--early_stop_patience 15 \
--reduce_lr_factor 0.5 \
--reduce_lr_patience 7 \
--reduce_lr_min 0.00000625 \
--augmentation_name valid_plus \
--augmentation_prob 1.0 \
--network unet_resnext_50 \
--alias _lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder oof

# 
# python3 predict_oof.py \
# --epochs 30 \
# --pretrain_weights /home/branding_images/salt/unet_resnet_152_192_224_snapshot_100_epochs_bs_16/fold_{}.hdf5 \
# --n_snapshots 1 \
# --fold 0,1,2,3 \
# --learning_rate 0.00005 \
# --input_size 224 \
# --resize_size 192 \
# --batch_size 16 \
# --loss_function lovasz \
# --callback snapshot \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnet_152 \
# --alias _192_224_snapshot_100_epochs_bs_16_finetune_lovash_v1 \
# --prediction_weights snapshot_1_fold_{}.hdf5 \
# --prediction_folder oof


# python3 predict_oof.py \
# --epochs 20 \
# --pretrain_weights /home/RnD/babakhin.y/salt/unet_resnet_50_192_224_initial/fold_{}.hdf5 \
# --n_snapshots 1 \
# --fold 0,1,2,3,4 \
# --learning_rate 0.00005 \
# --input_size 224 \
# --resize_size 192 \
# --batch_size 32 \
# --loss_function lovasz \
# --callback snapshot \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnet_50 \
# --alias _lovasz_192_224_initial_finetune_lovash \
# --prediction_weights fold_{}.hdf5 \
# --prediction_folder oof

# python3 predict_oof.py \

# --prediction_weights fold_{}.hdf5 \
# --prediction_folder oof

# python3 predict_oof.py \

# --prediction_weights snapshot-1.h5 \
# --prediction_folder oof