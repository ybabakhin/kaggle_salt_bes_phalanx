
python3 predict_oof.py \
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
--network unet_resnet_34 \
--alias _lovasz_101_128_on_pseudo_cyclic_1 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder oof

python3 predict_oof.py \
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
--network unet_resnet_34 \
--alias _lovasz_101_128_on_pseudo_cyclic_2 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder oof


python3 predict_oof.py \
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
--network unet_resnet_34 \
--alias _lovasz_101_128_on_pseudo_cyclic_3 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder oof


python3 predict_oof.py \
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
--network unet_resnet_34 \
--alias _lovasz_101_128_on_pseudo_cyclic_4 \
--prediction_weights fold_{}.hdf5 \
--prediction_folder oof



# python3 predict_oof.py \

# --prediction_weights snapshot-1.h5 \
# --prediction_folder oof