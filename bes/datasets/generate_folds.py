import os
import pandas as pd
import numpy as np
import cv2
from params import args

if __name__ == '__main__':
    n_fold = 5
    depths = pd.read_csv(os.path.join('../', args.data_root, 'depths.csv'))
    depths.sort_values('z', inplace=True)
    depths.drop('z', axis=1, inplace=True)
    depths['fold'] = (list(range(n_fold)) * depths.shape[0])[:depths.shape[0]]
    df_train = pd.read_csv(os.path.join('../', args.data_root, 'train.csv'))

    df_train = df_train.merge(depths)

    dist = []
    for id in df_train.id.values:
        img = cv2.imread(os.path.join('../', args.images_dir, '{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
        dist.append(np.unique(img).shape[0])
    df_train['unique_pixels'] = dist

    df_train[['id', 'fold', 'unique_pixels']].sample(frac=1, random_state=123).to_csv(
        os.path.join('../', args.data_root, 'train_proc_v2_gr.csv'), index=False)
