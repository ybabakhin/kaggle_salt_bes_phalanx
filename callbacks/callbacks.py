from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from callbacks.clr_callback import CyclicLR
from callbacks.snapshot import SnapshotCallbackBuilder
from params import args
import os


def get_callback(callback, **kwargs):
    if callback == 'early_stopping':
        es_callback = EarlyStopping(monitor="val_lb_metric", patience=args.early_stop_patience, mode='max')
        callbacks = [es_callback]

    elif callback == 'scheduler':

        def step_decay_schedule(initial_lr=1e-3):
            '''
            Wrapper function to create a LearningRateScheduler with step decay schedule.
            '''

            def schedule(epoch):
                if epoch <= 15:
                    return initial_lr
                elif epoch <= 15 + 40:
                    return initial_lr / 2
                elif epoch <= 15 + 40 + 60:
                    return initial_lr / 10
                else:
                    return initial_lr / 20

            return LearningRateScheduler(schedule)

        lr_sched = step_decay_schedule(initial_lr=1e-4)

        callbacks = [lr_sched]

    elif callback == 'reduce_lr':
        es_callback = EarlyStopping(monitor="val_lb_metric", patience=args.early_stop_patience, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_lb_metric', factor=args.reduce_lr_factor,
                                      patience=args.reduce_lr_patience,
                                      min_lr=args.reduce_lr_min, verbose=1, mode='max')

        callbacks = [es_callback, reduce_lr]

    elif callback == 'cyclic_lr':
        es_callback = EarlyStopping(monitor="val_loss", patience=args.early_stop_patience, min_delta=0.0005)
        clr_triangular = CyclicLR(mode='triangular2', base_lr=args.learning_rate / 16, max_lr=args.learning_rate / 4,
                                  step_size=2000)
        callbacks = [es_callback, clr_triangular]

    elif callback == 'snapshot':
        snapshot = SnapshotCallbackBuilder(kwargs['weights_path'], args.epochs, args.n_snapshots, args.learning_rate)
        callbacks = snapshot.get_callbacks(model_prefix='snapshot', fold=kwargs['fold'])

    else:
        ValueError("Unknown callback")

    mc_callback_best = ModelCheckpoint(kwargs['weights_path'], monitor='val_lb_metric', verbose=0, save_best_only=True,
                                       save_weights_only=True, mode='max', period=1)
    callbacks.append(mc_callback_best)

    tensorboard_dir = os.path.join(args.models_dir, 'logs/', '_'.join(kwargs['weights_path'].split('/')[-2:]))
    tensorboard_callback = TensorBoard(log_dir=tensorboard_dir)

    callbacks.append(tensorboard_callback)

    return callbacks