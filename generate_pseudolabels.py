import os
import pandas as pd
import numpy as np
import cv2

if __name__ == '__main__':
    from utils import generate_pseudolabels

    models = [

        'unet_resnext_50_lovasz_192_224_on_pseudo_000005_cyclic_1_upd',
        'unet_resnext_50_lovasz_192_224_on_pseudo_000005_cyclic_2_upd',
        'unet_resnext_50_lovasz_192_224_on_pseudo_000005_cyclic_3_upd',
        'unet_resnext_50_lovasz_192_224_on_pseudo_000005'

    ]
    model_pathes = ['/home/RnD/babakhin.y/salt/' + x for x in models]

    pseudolabels = generate_pseudolabels(model_pathes, [0, 1, 2, 3, 4], test.id.values, 0.5, phalanx_dict,
                                         weights=[1, 1, 1, 3])

    pseudolabels_df = pd.DataFrame(pseudolabels).T
    pseudolabels_df.columns = ['confidence', 'mask']

    DATA_ROOT = 'data/'

    df_test = pd.read_csv(os.path.join(DATA_ROOT, 'sample_submission.csv'), index_col='id')
    depths = pd.read_csv(os.path.join(DATA_ROOT, 'depths.csv'), index_col='id')
    df_test = df_test.join(depths)
    # Label suspicious images
    dist = []
    mask_pixels = []
    for id in df_test.index:
        img = cv2.imread(os.path.join('data/test/images/', '{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
        mask = pseudolabels_df.loc[id][1]
        dist.append(np.unique(img).shape[0])
        mask_pixels.append(np.sum(mask / 255))
    df_test['unique_pixels'] = dist
    df_test['mask_pixels'] = mask_pixels

    df_test = df_test.join(pseudolabels_df)

    for idx, row in df_test.iterrows():
        cv2.imwrite(
            os.path.join('/home/p/babakhin/Branding/salt_old/kaggle-salt/data/pseudolabels_v2/', str(idx) + '.png'),
            np.array(row['mask'], np.uint8))

    df_test[['unique_pixels', 'mask_pixels', 'confidence']].sample(frac=1, random_state=123).to_csv(
        '/home/p/babakhin/Branding/salt_old/kaggle-salt/pseudolabels_v2.csv')

    pseudo = df_test[(df_test.confidence >= 0.8) & (df_test.unique_pixels > 1)].sample(frac=1, random_state=123)
    pseudo['coverage_class'] = -1
    pseudo['fold'] = -1
    pseudo[['fold', 'unique_pixels', 'coverage_class']].to_csv(
        '/home/p/babakhin/Branding/salt_old/kaggle-salt/data/pseudolabels_v2.csv')
