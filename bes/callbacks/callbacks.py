from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from callbacks.snapshot import SnapshotCallbackBuilder
from params import args


def get_callback(callback, **kwargs):
    if callback == 'reduce_lr':
        es_callback = EarlyStopping(monitor="val_lb_metric", patience=args.early_stop_patience, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_lb_metric', factor=args.reduce_lr_factor,
                                      patience=args.reduce_lr_patience,
                                      min_lr=args.reduce_lr_min, verbose=1, mode='max')

        callbacks = [es_callback, reduce_lr]
    elif callback == 'snapshot':
        snapshot = SnapshotCallbackBuilder(kwargs['weights_path'], args.epochs, args.n_snapshots, args.learning_rate)
        callbacks = snapshot.get_callbacks(model_prefix='snapshot', fold=kwargs['fold'])
    else:
        ValueError("Unknown callback")

    mc_callback_best = ModelCheckpoint(kwargs['weights_path'], monitor='val_lb_metric', verbose=0, save_best_only=True,
                                       save_weights_only=True, mode='max', period=1)
    callbacks.append(mc_callback_best)

    return callbacks
