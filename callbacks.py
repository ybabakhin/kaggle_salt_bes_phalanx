from keras.callbacks import *
from clr_callback import CyclicLR
from params import args
import os


from io import BytesIO
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
import matplotlib.pyplot as plt

from snapshot import SnapshotCallbackBuilder

class TensorBoardPrediction(Callback):
    """A TensorBoard callback to display samples, targets, and predictions.

    Args:
        generator (keras.utils.Sequence): A data generator to iterate over the
            dataset.
        class_to_rgb (OrderedDict): An ordered dictionary that relates pixel
            values, class names, and class colors.
        log_dir (string): Specifies the directory where TensorBoard will
            write TensorFlow event files that it can display.
        batch_index (int): The batch index to display. Default: 0.
        max_outputs (int): Max number of elements in the batch to generate
            images for. Default: 3.

    """

    def __init__(
        self, generator, log_dir, batch_index=0, max_outputs=3
    ):
        super().__init__()

        self.generator = generator
        self.batch_index = batch_index
        self.max_outputs = max_outputs
        self.log_dir = log_dir

    def set_model(self, model):
        """Creates a FileWriter to write the TensorBoard event file."""
        super().set_model(model)
        self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, _):
        """Updates the TensorBoard event file with images.

        Computes the current model prediction on a batch and adds the
        sample, target, and prediction images to TensorBoard.

        Args:
            epoch (int): Current epoch.

        """
        sample, y_true = self.generator[self.batch_index]
        y_pred = np.array(self.model.predict_on_batch(sample))

        y_true = np.array(y_true * 255, np.uint8)
        y_pred = np.array(y_pred * 255, np.uint8)
        
        batch_summary = self.image_summary(sample, 'sample')
        batch_summary += self.image_summary(y_true, 'target')
        batch_summary += self.image_summary(y_pred, 'prediction')
        summary = tf.Summary(value=batch_summary)

        # Write the summaries to the file
        self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        """Close the TensorBoard FileWriter."""
        self.writer.close()

    def image_summary(self, batch, tag):
        """Constructs a list of image summaries for a batch of images.

        Args:
            batch (numpy.ndarray): A batch of images to generate TensorBaord
                summaries for.
            tag (string): The name to identify the images. In TensorBoard,
                tags are often organized by scope (which contains slashes
                to convey hierarchy).

        Returns:
            A list of TensorBoard summaries with images.

        """
        assert batch.shape[-1] == 3, (
            "expected image with 3 channels got {}".format(batch.shape[-1])
        )

        # If batch is actually just a single image with 3 dimensions give it
        # a batch dimension equal to 1
        if np.ndim(batch) == 3:
            batch = np.expand_dims(batch, 0)

        # Dimensions
        batch_size, height, width, channels = batch.shape

        if self.max_outputs > batch_size:
            self.max_outputs = batch_size

        summary_list = []
        for idx in range(0, self.max_outputs):
            image = batch[idx]

            # We need the images in encoded format (bytes); to get that we
            # must save it to a byte stream...
            image_io = BytesIO()
            plt.imsave(image_io, image, format='png')

            # ...and get its contents after
            image_string_io = image_io.getvalue()
            image_io.close()

            # Create and append the summary to the list
            image_summary = tf.Summary.Image(
                height=height,
                width=width,
                colorspace=channels,
                encoded_image_string=image_string_io
            )
            image_tag = "{0}/{1}".format(tag, idx + 1)
            summary_list.append(
                tf.Summary.Value(tag=image_tag, image=image_summary)
            )

        return summary_list

def get_callback(callback, **kwargs):
    if callback == 'early_stopping':
        es_callback = EarlyStopping(monitor="val_loss", patience=args.early_stop_patience, min_delta=0.0005)
        #es_callback = EarlyStopping(monitor="val_competitionMetric2", patience=args.early_stop_patience, mode='max')
        
        callbacks = [es_callback]
        
    elif callback == 'scheduler':
        
        def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
            '''
            Wrapper function to create a LearningRateScheduler with step decay schedule.
            '''
            def schedule(epoch):
                
                # Train 200
#                 if epoch <= 15:
#                     return initial_lr
#                 elif epoch <= 15+40:
#                     return initial_lr/2
#                 elif epoch <= 15+40+60:
#                     return initial_lr/10
#                 else:
#                     return initial_lr/20
                
                # Train 150
#                 if epoch <= 15:
#                     return initial_lr
#                 elif epoch <= 15+40:
#                     return initial_lr/2
#                 elif epoch <= 15+40+40:
#                     return initial_lr/10
#                 else:
#                     return initial_lr/20
                
                # Finetune
                if epoch <= 15:
                    return initial_lr/10
                elif epoch <= 15+15:
                    return initial_lr/20
                else:
                    return initial_lr/100

            return LearningRateScheduler(schedule)

        lr_sched = step_decay_schedule(initial_lr=1e-4)


        callbacks = [lr_sched]


    elif callback == 'reduce_lr':
        es_callback = EarlyStopping(monitor="val_lb_metric", patience=args.early_stop_patience, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_lb_metric',factor=args.reduce_lr_factor, patience=args.reduce_lr_patience,
                                      min_lr=args.reduce_lr_min, verbose=1,mode='max')

        callbacks = [es_callback, reduce_lr]

    elif callback == 'cyclic_lr':
        es_callback = EarlyStopping(monitor="val_loss", patience=args.early_stop_patience, min_delta=0.0005)

        # Cyclic LR
        clr_triangular = CyclicLR(mode='triangular2', base_lr=args.learning_rate/16, max_lr=args.learning_rate/4, step_size=2000)
        callbacks = [es_callback, clr_triangular]
        
        
    elif callback == 'snapshot':
        snapshot = SnapshotCallbackBuilder(kwargs['weights_path'], args.epochs, args.n_snapshots, args.learning_rate)
        callbacks = snapshot.get_callbacks(model_prefix='snapshot', fold=kwargs['fold'])

    else:
        ValueError("Unknown callback")

    mc_callback_best = ModelCheckpoint(kwargs['weights_path'], monitor='val_lb_metric', verbose=0, save_best_only=True,
                                  save_weights_only=True, mode='max', period=1)
    callbacks.append(mc_callback_best)
    
#     mc_callback_all = ModelCheckpoint(kwargs['weights_path']+'.{epoch:03d}-{val_loss:.5f}-{val_lb_metric:.5f}.hdf5', monitor='val_lb_metric', verbose=0, save_best_only=False,
#                                   save_weights_only=True, mode='max', period=1)
#     callbacks.append(mc_callback_all)
    
    tensorboard_dir = os.path.join(args.models_dir,'logs/','_'.join(kwargs['weights_path'].split('/')[-2:]))
    tensorboard_callback = TensorBoard(log_dir=tensorboard_dir)
    
    callbacks.append(tensorboard_callback)
    
#     tensorboard_viz = TensorBoardPrediction(
#         kwargs['val_generator'],
#         log_dir=tensorboard_dir
# )
#     callbacks.append(tensorboard_viz)

    return callbacks