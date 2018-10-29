# Pseudolabels
# python3 train.py \
# --epochs 40 \
# --n_snapshots 1 \
# --fold 0 \
# --learning_rate 0.0001 \
# --input_size 128 \
# --resize_size 101 \
# --batch_size 128 \
# --loss_function bce_dice \
# --callback snapshot \
# --freeze_encoder 0 \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnet_34 \
# --pseudolabels_dir /home/p/babakhin/Branding/salt_old/kaggle-salt/data/pseudolabels_v2/ \
# --alias _101_128_pseudolabels_v2

# python3 train.py \
# --epochs 40 \
# --n_snapshots 1 \
# --pretrain_weights /home/RnD/babakhin.y/salt/unet_resnet_34_101_128_pseudolabels_v2/fold_0.hdf5 \
# --fold 0 \
# --learning_rate 0.00005 \
# --input_size 128 \
# --resize_size 101 \
# --batch_size 128 \
# --loss_function lovasz \
# --callback snapshot \
# --freeze_encoder 0 \
# --augmentation_name valid_plus \
# --augmentation_prob 1.0 \
# --network unet_resnet_34_lovasz \
# --pseudolabels_dir /home/p/babakhin/Branding/salt_old/kaggle-salt/data/pseudolabels_v2/ \
# --alias _101_128_pseudolabels_v2

# python3 train.py \
# --epochs 125 \
# --pretrain_weights /home/RnD/babakhin.y/salt/unet_resnet_34_lovasz_101_128_pseudolabels_v2/fold_0.hdf5 \
# --fold 0,1,2,3,4 \
# --learning_rate 0.00005 \
# --input_size 128 \
# --resize_size 101 \
# --batch_size 32 \
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
# --alias _101_128_on_pseudo_000005


python3 train.py \
--epochs 25 \
--n_snapshots 1 \
--pretrain_weights /home/RnD/babakhin.y/salt/unet_resnet_34_lovasz_101_128_on_pseudo_000005/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.00005 \
--input_size 128 \
--resize_size 101 \
--batch_size 32 \
--loss_function lovasz \
--callback snapshot \
--freeze_encoder 0 \
--augmentation_name valid \
--augmentation_prob 1.0 \
--network unet_resnet_34_lovasz \
--alias _101_128_on_pseudo_cyclic_1

python3 train.py \
--epochs 25 \
--n_snapshots 1 \
--pretrain_weights /home/RnD/babakhin.y/salt/unet_resnet_34_lovasz_101_128_on_pseudo_cyclic_1/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.00005 \
--input_size 128 \
--resize_size 101 \
--batch_size 32 \
--loss_function lovasz \
--callback snapshot \
--freeze_encoder 0 \
--augmentation_name valid \
--augmentation_prob 1.0 \
--network unet_resnet_34_lovasz \
--alias _101_128_on_pseudo_cyclic_2

python3 train.py \
--epochs 25 \
--n_snapshots 1 \
--pretrain_weights /home/RnD/babakhin.y/salt/unet_resnet_34_lovasz_101_128_on_pseudo_cyclic_2/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.00005 \
--input_size 128 \
--resize_size 101 \
--batch_size 32 \
--loss_function lovasz \
--callback snapshot \
--freeze_encoder 0 \
--augmentation_name valid \
--augmentation_prob 1.0 \
--network unet_resnet_34_lovasz \
--alias _101_128_on_pseudo_cyclic_3

python3 train.py \
--epochs 25 \
--n_snapshots 1 \
--pretrain_weights /home/RnD/babakhin.y/salt/unet_resnet_34_lovasz_101_128_on_pseudo_cyclic_3/fold_{}.hdf5 \
--fold 0,1,2,3,4 \
--learning_rate 0.00005 \
--input_size 128 \
--resize_size 101 \
--batch_size 32 \
--loss_function lovasz \
--callback snapshot \
--freeze_encoder 0 \
--augmentation_name valid \
--augmentation_prob 1.0 \
--network unet_resnet_34_lovasz \
--alias _101_128_on_pseudo_cyclic_4


