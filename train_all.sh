# python3 train.py \
# --epochs 50 \
# --n_snapshots 1 \
# --fold 0 \
# --learning_rate 0.0001 \
# --input_size 224 \
# --resize_size 192 \
# --batch_size 24 \
# --loss_function bce_jacard \
# --callback snapshot \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnet_50_do1 \
# --alias _exp_0_192_224_snapshot_50_epochs_qubvel_auto_builder_fix_bn


# python3 train.py \
# --epochs 50 \
# --n_snapshots 1 \
# --fold 0 \
# --learning_rate 0.0001 \
# --input_size 224 \
# --resize_size 192 \
# --batch_size 24 \
# --loss_function bce_jacard \
# --callback snapshot \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnet_50_do2 \
# --alias _exp_0_192_224_snapshot_50_epochs_dec_capacity_hypercolumn


python3 train.py \
--epochs 30 \
--pretrain_weights /home/branding_images/salt/unet_resnet_152_192_224_snapshot_100_epochs_bs_16/fold_{}.hdf5 \
--n_snapshots 1 \
--fold 4 \
--learning_rate 0.00005 \
--input_size 224 \
--resize_size 192 \
--batch_size 16 \
--loss_function lovasz \
--callback snapshot \
--augmentation_name valid_plus \
--augmentation_prob 1.0 \
--network unet_resnet_152 \
--alias _192_224_snapshot_100_epochs_bs_16_finetune_lovash_v1


# --pretrain_weights /home/branding_images/salt/unet_resnet_152_exp_0_202_256_scheduler_150_epochs/fold_{}.hdf5