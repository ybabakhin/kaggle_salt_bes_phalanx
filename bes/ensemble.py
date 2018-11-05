import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from params import args


# Ensemble parameters:
# List of models, list of weights, threshold, postprocessing?, save pseudolabels?, submission filename?

# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


def read_phalanx_test(path):
    phalanx = np.load(path)

    min_prob = float("inf")
    max_prob = -float("inf")
    for m in phalanx:
        mi = np.min(m)
        ma = np.max(m)
        if mi < min_prob:
            min_prob = mi
        if ma > max_prob:
            max_prob = ma

    train_id = pd.read_csv(os.path.join(args.data_root, 'train.csv'))['id'].values
    depth_id = pd.read_csv(os.path.join(args.data_root, 'depths.csv'))['id'].values
    test_id = np.setdiff1d(depth_id, train_id)

    phalanx_df = pd.read_csv(os.path.join(args.data_root, 'sample_submission.csv'))
    phalanx_df['id'] = test_id

    phalanx_dict = {}
    for idx, row in phalanx_df.iterrows():
        phalanx_dict[row['id']] = (phalanx[idx] - min_prob) / (max_prob - min_prob)

    return phalanx_dict


def ensemble(model_dirs, folds, ids, thr, phalanx_dicts=None, weights=None, inner_weights=None):
    rles = []
    pseudolabels = {}

    if weights is None:
        weights = [1] * len(model_dirs)

    for img_id in tqdm(ids):
        preds = []
        for d, w in zip(model_dirs, weights):
            pred_folds = []
            for fold in folds:
                path = os.path.join(d, 'fold_{}'.format(fold))
                mask = cv2.imread(os.path.join(path, '{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)

                img = cv2.imread(os.path.join(args.test_folder, '{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)
                if np.unique(img).shape[0] == 1:
                    pred_folds.append(np.zeros(mask.shape))
                else:
                    pred_folds.append(np.array(mask / 255, np.float32))
            preds.append(np.mean(np.array(pred_folds, np.float32), axis=0) * w)

        if phalanx_dicts is None:
            final_pred = np.sum(np.array(preds), axis=0) / sum(weights)
        else:

            if len(model_dirs) == 0:
                final_pred = []
            else:
                final_pred = [np.sum(np.array(preds), axis=0) / sum(weights) * inner_weights[0]]

            i = 1
            for phalanx_dict in phalanx_dicts:
                final_pred.append(phalanx_dict[img_id] * inner_weights[i])
                i += 1
            final_pred = np.sum(np.array(final_pred), axis=0) / sum(inner_weights)

        confidence = (np.sum(final_pred < 0.2) + np.sum(final_pred > 0.8)) / (101 ** 2)
        mask = final_pred > thr
        pseudolabels[img_id] = (confidence, mask * 255)

        rle = RLenc(mask)
        rles.append(rle)

    return rles, pseudolabels


def generate_pseudolabels(pseudolabels, pseudolabels_path='pseudolabels'):
    pseudolabels_df = pd.DataFrame(pseudolabels).T
    pseudolabels_df.columns = ['confidence', 'mask']

    df_test = pd.read_csv(os.path.join(args.data_root, 'sample_submission.csv'), index_col='id')
    depths = pd.read_csv(os.path.join(args.data_root, 'depths.csv'), index_col='id')
    df_test = df_test.join(depths)
    # Label suspicious images
    dist = []
    mask_pixels = []
    for id in df_test.index:
        img = cv2.imread(os.path.join(args.test_folder, '{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
        mask = pseudolabels_df.loc[id][1]
        dist.append(np.unique(img).shape[0])
        mask_pixels.append(np.sum(mask / 255))
    df_test['unique_pixels'] = dist
    df_test['mask_pixels'] = mask_pixels

    df_test = df_test.join(pseudolabels_df)

    for idx, row in df_test.iterrows():
        cv2.imwrite(
            os.path.join(args.data_root, pseudolabels_path, str(idx) + '.png'),
            np.array(row['mask'], np.uint8))

    df_test[['unique_pixels', 'mask_pixels', 'confidence']].sample(frac=1, random_state=123).to_csv(
        os.path.join(args.data_root, pseudolabels_path + '.csv'))

    pseudo = df_test[(df_test.confidence >= 0.9) & (df_test.unique_pixels > 1)].sample(frac=1, random_state=123)
    pseudo['fold'] = -1
    pseudo[['fold', 'unique_pixels']].to_csv(
        os.path.join(args.data_root, pseudolabels_path + '_confident.csv'))


def postprocessing(test):
    df_train = pd.read_csv(os.path.join(args.data_root, 'train.csv'))

    masks = []
    dist = []

    for id in df_train.id.values:
        mask = cv2.imread(os.path.join(args.masks_dir, '{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
        masks.append(np.array(mask) / 255.)
        img = cv2.imread(os.path.join(args.images_dir, '{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
        dist.append(np.unique(img).shape[0])

    df_train['unique_pixels'] = dist
    df_train['masks'] = masks

    df_train = df_train[df_train.unique_pixels > 1]

    def get_mask_type(mask):
        if ((mask[-50:, :]).sum() > 0) and np.all(mask[-50:, :] == mask[-1]) and np.any(mask[-1] != 1):
            return 1
        else:
            return 0

    df_train["is_vertical_soft"] = df_train.masks.map(get_mask_type)

    def get_mask_type(mask):
        if (mask.sum() > 0) and np.all(mask == mask[-1]) and np.any(mask[-1] != 1):
            return 1
        else:
            return 0

    df_train["is_vertical"] = df_train.masks.map(get_mask_type)

    verts_dict = {}
    for mos_csv in os.listdir(os.path.join(args.data_root, 'mos_numpy_v2/')):
        mos = pd.read_csv(os.path.join(args.data_root, 'mos_numpy_v2/', mos_csv), header=None)
        for col in range(mos.shape[1]):
            for idx, row in mos.iterrows():
                ans_down = None
                ans_up = None
                if row[col] in list(df_train[df_train["is_vertical"] == 1].id.values):
                    new_mask = np.array([list(df_train[df_train.id == row[col]].masks.values[0][-1]), ] * 101)
                    ans_down = RLenc(new_mask)
                    new_mask = np.array([list(df_train[df_train.id == row[col]].masks.values[0][0]), ] * 101)
                    ans_up = RLenc(new_mask)
                    break
                elif row[col] in list(df_train[df_train["is_vertical_soft"] == 1].id.values):
                    new_mask = np.array([list(df_train[df_train.id == row[col]].masks.values[0][-1]), ] * 101)
                    ans_down = RLenc(new_mask)
                    break
            if ans_down is not None:
                for id in mos.loc[idx + 1:, col]:
                    verts_dict[id] = ans_down
            if ans_up is not None and idx >= 3:
                for id in mos.loc[idx - 1:idx, col]:
                    verts_dict[id] = ans_up

    final_pred = []
    for idx, row in test.iterrows():
        if row['id'] in verts_dict.keys():
            final_pred.append(verts_dict[row['id']])
        else:
            final_pred.append(row['rle_mask'])
    test['rle_mask'] = final_pred

    return test


if __name__ == '__main__':
    test_ids = [x[:-4] for x in os.listdir(args.test_folder) if x[-4:] == '.png']
    print('Generating Predictions for Stage', args.stage)
    if args.stage == 1:
        models = [
            'unet_resnext_50_lovasz_stage_1_5',
            'unet_resnext_50_lovasz_stage_1_4',
            'unet_resnext_50_lovasz_stage_1_3',
            'unet_resnext_50_lovasz_stage_1_2'
        ]

        model_pathes = [args.models_dir + x for x in models]
        phalanx_dicts = [read_phalanx_test('/workdir/phalanx/predictions/phalanx_stage_1.npy')]

        pred, pseudolabels = ensemble(model_pathes, [0, 1, 2, 3, 4],
                                      test_ids, 0.5,
                                      phalanx_dicts=phalanx_dicts,
                                      weights=[1, 1, 1, 1],
                                      inner_weights=[1, 1])

        generate_pseudolabels(pseudolabels, 'pseudolabels')

    elif args.stage == 2:
        models = [
            'unet_resnext_50_lovasz_stage_2_5',
            'unet_resnext_50_lovasz_stage_2_4',
            'unet_resnext_50_lovasz_stage_2_3',
            'unet_resnext_50_lovasz_stage_2_2'
        ]

        model_pathes = [args.models_dir + x for x in models]
        phalanx_dicts = [read_phalanx_test('/workdir/phalanx/predictions/phalanx_stage_2.npy')]

        pred, pseudolabels = ensemble(model_pathes, [0, 1, 2, 3, 4],
                                      test_ids, 0.5,
                                      phalanx_dicts=phalanx_dicts,
                                      weights=[1, 1, 1, 3],
                                      inner_weights=[1, 1])

        generate_pseudolabels(pseudolabels, 'pseudolabels_v2')

    elif args.stage == 3:
        models = [
            'unet_resnext_50_lovasz_stage_2_5',
            'unet_resnext_50_lovasz_stage_2_4',
            'unet_resnext_50_lovasz_stage_2_3',
            'unet_resnext_50_lovasz_stage_2_2'
        ]

        model_pathes = [args.models_dir + x for x in models]
        phalanx_dicts = [read_phalanx_test('/workdir/phalanx/predictions/phalanx_stage_3.npy')]

        pred, pseudolabels = ensemble(model_pathes, [0, 1, 2, 3, 4],
                                      test_ids, 0.5,
                                      phalanx_dicts=phalanx_dicts,
                                      weights=[1, 1, 1, 3],
                                      inner_weights=[1, 1])

        # Whether to generate test predictions?
        # generate_pseudolabels(pseudolabels, 'test_predictions')

        test = pd.DataFrame({'id': test_ids, 'rle_mask': pred})

        if args.postprocessing == 1:
            print('Applying Postrpocessing...')
            test = postprocessing(test)

        test[['id', 'rle_mask']].to_csv(args.test_predictions_path, index=False)
