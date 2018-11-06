import pandas as pd
import numpy as np
import os
import gc
from keras import backend as K
from keras.optimizers import RMSprop
from utils import predict_test
from params import args
from models.models import get_model
from losses import Kaggle_IoU_Precision, make_loss


def main():
    test_ids = np.array([x[:-4] for x in os.listdir(args.test_folder) if x[-4:] == '.png'])

    MODEL_PATH = os.path.join(args.models_dir, args.network + args.alias)
    folds = [int(f) for f in args.fold.split(',')]

    print('Predicting Model:', args.network + args.alias)

    for fold in folds:
        K.clear_session()
        print('***************************** FOLD {} *****************************'.format(fold))

        # Initialize Model
        weights_path = os.path.join(MODEL_PATH, args.prediction_weights.format(fold))

        model, preprocess = get_model(args.network,
                                      input_shape=(args.input_size, args.input_size, 3),
                                      freeze_encoder=args.freeze_encoder)
        model.compile(optimizer=RMSprop(lr=args.learning_rate), loss=make_loss(args.loss_function),
                      metrics=[Kaggle_IoU_Precision])

        model.load_weights(weights_path)

        # Save test predictions to disk
        dir_path = os.path.join(MODEL_PATH, args.prediction_folder.format(fold))
        os.system("mkdir {}".format(dir_path))
        predict_test(model=model,
                     preds_path=dir_path,
                     ids=test_ids,
                     batch_size=args.batch_size * 2,
                     TTA='flip',
                     preprocess=preprocess)

        gc.collect()


if __name__ == '__main__':
    main()
