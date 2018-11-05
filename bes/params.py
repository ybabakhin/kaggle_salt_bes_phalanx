import argparse
import sys

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--epochs', type=int, default=50)
arg('--n_snapshots', type=int, default=1)
arg('--fold', default='0')
arg('--pretrain_weights')
arg('--prediction_weights', default='fold_{}.hdf5')
arg('--prediction_folder', default='oof')
arg('--learning_rate', type=float, default=0.0001)
arg('--input_size', type=int, default=192)
arg('--resize_size', type=int, default=160)
arg('--batch_size', type=int, default=24)

arg('--loss_function', default='bce_jacard')
arg('--augmentation_name', default='valid')
arg('--augmentation_prob', type=float, default=1.0)
arg('--network', default='unet_resnet_50')
arg('--alias', default='')
arg('--callback', default='snapshot')
arg('--freeze_encoder', type=int, default=0)

arg('--models_dir', default='weights/')
arg('--data_root', default='../data/')
arg('--images_dir', default='../data/train/images/')
arg('--pseudolabels_dir', default='')
arg('--masks_dir', default='../data/train/masks/')
arg('--test_folder', default='../data/test/images/')
arg('--folds_csv', default='../data/train_proc_v2_gr.csv')
arg('--pseudolabels_csv', default='../data/pseudolabels_confident.csv')

# arg('--models_dir', default='/workdir/bes/weights/')
# arg('--data_root', default='/workdir/bes/data/')
# arg('--images_dir', default='/workdir/bes/data/train/images/')
# arg('--pseudolabels_dir', default='')
# arg('--masks_dir', default='/workdir/bes/data/train/masks/')
# arg('--test_folder', default='/workdir/bes/data/test/images/')
# arg('--folds_csv', default='/workdir/bes/data/train_proc_v2_gr.csv')
# arg('--pseudolabels_csv', default='/workdir/bes/data/pseudolabels_confident.csv')

arg('--initial_size', type=int, default=101)
arg('--num_workers', type=int, default=12)
arg('--early_stop_patience',  type=int, default=15)
arg('--reduce_lr_factor',  type=float, default=0.25)
arg('--reduce_lr_patience',  type=int, default=7)
arg('--reduce_lr_min',  type=float, default=0.000001)

arg('--stage',  type=int, default=3)
arg('--postprocessing',  type=int, default=0)

args = parser.parse_args()
