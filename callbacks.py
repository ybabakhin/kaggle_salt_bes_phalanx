from keras.callbacks import *
from clr_callback import CyclicLR


def get_callback(callback, **kwargs):
    if callback == 'early_stopping':
        es_callback = EarlyStopping(monitor="val_loss", patience=kwargs['early_stop_patience'])

        callbacks = [es_callback]

    elif callback == 'reduce_lr':
        es_callback = EarlyStopping(monitor="val_loss", patience=kwargs['early_stop_patience'])
        reduce_lr = ReduceLROnPlateau(factor=kwargs['reduce_lr_factor'], patience=kwargs['reduce_lr_patience'],
                                      min_lr=kwargs['reduce_lr_min'], verbose=1)
        callbacks = [es_callback, reduce_lr]

    elif callback == 'cyclic_lr':
        es_callback = EarlyStopping(monitor="val_loss", patience=kwargs['early_stop_patience'])
        reduce_lr = ReduceLROnPlateau(factor=kwargs['reduce_lr_factor'], patience=kwargs['reduce_lr_patience'],
                                      min_lr=kwargs['reduce_lr_min'], verbose=1)

        # Cyclic LR
        clr_triangular = CyclicLR(mode='triangular', base_lr=0.001, max_lr=0.006, step_size=2000.)
        callbacks = [es_callback, reduce_lr, clr_triangular]

    else:
        ValueError("Unknown callback")

    mc_callback = ModelCheckpoint(kwargs['weights_path'], monitor='val_loss', verbose=0, save_best_only=True,
                                  save_weights_only=True, mode='auto', period=1)
    callbacks.append(mc_callback)

    return callbacks