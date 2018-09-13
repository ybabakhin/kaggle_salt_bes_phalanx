python3 train.py \
--epochs 50 \
--fold 0 \
--learning_rate 0.0001 \
--input_size 192 \
--resize_size 144 \
--batch_size 24 \
--loss_function bce_jacard \
--callback snapshot \
--augmentation_name valid_plus_v1 \
--augmentation_prob 1.0 \
--network unet_resnet_50 \
--alias _exp_0_144_192_snapshot_50_epochs_vp_1

python3 train.py \
--epochs 50 \
--fold 0 \
--learning_rate 0.0001 \
--input_size 192 \
--resize_size 144 \
--batch_size 24 \
--loss_function bce_jacard \
--callback snapshot \
--augmentation_name valid_plus_v2 \
--augmentation_prob 1.0 \
--network unet_resnet_50 \
--alias _exp_0_144_192_snapshot_50_epochs_vp_2

python3 train.py \
--epochs 50 \
--fold 0 \
--learning_rate 0.0001 \
--input_size 192 \
--resize_size 144 \
--batch_size 24 \
--loss_function bce_jacard \
--callback snapshot \
--augmentation_name valid_plus_v3 \
--augmentation_prob 1.0 \
--network unet_resnet_50 \
--alias _exp_0_144_192_snapshot_50_epochs_vp_3

python3 train.py \
--epochs 50 \
--fold 0 \
--learning_rate 0.0001 \
--input_size 192 \
--resize_size 144 \
--batch_size 24 \
--loss_function bce_jacard \
--callback snapshot \
--augmentation_name valid_plus_v4 \
--augmentation_prob 1.0 \
--network unet_resnet_50 \
--alias _exp_0_144_192_snapshot_50_epochs_vp_4

python3 train.py \
--epochs 50 \
--fold 0 \
--learning_rate 0.0001 \
--input_size 192 \
--resize_size 144 \
--batch_size 24 \
--loss_function bce_jacard \
--callback snapshot \
--augmentation_name valid_plus_v5 \
--augmentation_prob 1.0 \
--network unet_resnet_50 \
--alias _exp_0_144_192_snapshot_50_epochs_vp_5


# python3 predict_oof.py \

# --prediction_weights fold_{}.hdf5 \
# --prediction_folder oof

# python3 predict_oof.py \

# --prediction_weights snapshot-1.h5 \
# --prediction_folder oof


# --pretrain_weights /home/branding_images/salt/unet_resnet_152_exp_0_202_256_scheduler_150_epochs/fold_{}.hdf5