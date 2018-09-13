import pandas as pd
import numpy as np
import os
from keras import backend as K
import gc
from keras.optimizers import Adam, RMSprop, SGD
from utils import predict_test, evaluate, ensemble, ThreadsafeIter, classification_predict_test
from params import args
from models.models import get_model
from losses import *


def main():
    print(args.alias)
    train = pd.read_csv(os.path.join(args.data_root, 'train_proc_v2.csv'))
    MODEL_PATH = os.path.join(args.models_dir, args.network + args.alias)
    folds = [int(f) for f in args.fold.split(',')]
    print(args.network + args.alias)
    
    for fold in folds:
        
        K.clear_session()
        print('***************************** FOLD {} *****************************'.format(fold))

        ids_valid = train[train.fold == fold].id.values

        # Initialize Model
        weights_path = os.path.join(MODEL_PATH, args.prediction_weights.format(fold))
        
        model, preprocess = get_model(args.network, input_shape=(args.input_size, args.input_size, 3), train_base=True)
        model.compile(optimizer=RMSprop(lr=args.learning_rate), loss=make_loss(args.loss_function),
                      metrics=[Kaggle_IoU_Precision])

        model.load_weights(weights_path)

        # SAVE OOF PREDICTIONS
        dir_path = os.path.join(MODEL_PATH, args.prediction_folder)
        os.system("mkdir {}".format(dir_path))
        pred = predict_test(model=model,
                            preds_path=dir_path,
                            oof=True,
                            ids=ids_valid,
                            batch_size=args.batch_size * 2,
                            thr=0.5,
                            TTA='flip',
                            preprocess=preprocess)

        
        res = evaluate([MODEL_PATH], train[train.fold.isin([fold])].id.values, 0.5, classification='')
        print("{} / {} / {}".format(np.round(np.mean(res['iout']),5),np.round(np.mean(res['dice']),5),np.round(np.mean(res['jacard']),5)))


        gc.collect()


if __name__ == '__main__':
    main()