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

    print(min_prob, max_prob)
    phalanx_df = pd.read_csv('res34_256_no_pseudo_real.csv')
    phalanx_dict = {}
    for idx, row in phalanx_df.iterrows():
        phalanx_dict[row['id']] = (phalanx[idx] - min_prob) / (max_prob - min_prob)
        # phalanx_dict[row['id']] = phalanx[idx]

    return phalanx_dict


def ensemble(model_dirs, folds, ids, thr, phalanx_dicts=None, weights=None, inner_weights=None):
    rles = []
    predicted_masks = {}
    predicted_probs = {}

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

        mask = final_pred > thr

        rle = RLenc(mask)
        rles.append(rle)

        predicted_probs[img_id] = final_pred
        predicted_masks[img_id] = mask * 255

    return rles, predicted_masks, predicted_probs


def postprocessing():
    # Vertical masks from train
    df_train = pd.read_csv('data/train.csv')

    masks = []
    dist = []

    for id in df_train.id.values:
        mask = cv2.imread(os.path.join('data/train/masks/', '{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
        masks.append(np.array(mask) / 255.)
        img = cv2.imread(os.path.join('data/train/images/', '{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
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

    # DO NOT BREAK IN A SINGLE COLUMN. MAYBE AVERAGE OF KNOWN PREDICTIONS?
    leaks_dict = {}
    for mos_csv in os.listdir('data/mos_numpy/'):
        mos = pd.read_csv(os.path.join('data/mos_numpy/', mos_csv), header=None)
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
                    leaks_dict[id] = ans_down
            if ans_up is not None and idx >= 3:
                for id in mos.loc[idx - 1:idx, col]:
                    leaks_dict[id] = ans_up

    changed = 0
    s_v = pd.DataFrame({'id': list(leaks_dict.keys()), 'hand_rle': list(leaks_dict.values())})
    print(s_v.shape)
    pred = []
    for idx, row in test.iterrows():
        if row['id'] in s_v.id.values:
            pred.append(s_v[s_v.id == row['id']]['hand_rle'].values[0])
            changed += 1
        else:
            pred.append(row['rle_mask'])
    print('Changed: ', changed)
    test['rle_mask'] = pred
    test[['id', 'rle_mask']].to_csv('6+12_89555_vertv1.csv', index=False)


if __name__ == '__main__':
    test = pd.read_csv(os.path.join(args.data_root, 'sample_submission.csv'))

    train = pd.read_csv(args.folds_csv)

    MODEL_PATH = '/home/RnD/babakhin.y/salt/pred_fold_res34_pad128_newpseudo_v2'

    models = [
        'unet_resnext_50_lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v1',
        'unet_resnext_50_lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v2',
        'unet_resnext_50_lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005_finetune_snapshots_low_thr_0001_v3',
        'unet_resnext_50_lovasz_192_224_reduce_lr_smaller_decoder_bce_dice_csse_usual_00005',
        # 'unet_resnet_152_192_224_snapshot_100_epochs_bs_16_finetune_lovash_v1'
    ]

    models = [
        # 'pred_fold_res34_pad128_newpseudo_v2',
        'pred_fold_res34_resize128_newpseudo_corr',
        # 'unet_resnext_50_lovasz_192_224_on_pseudo_000005_cyclic_1_upd',
        # 'unet_resnext_50_lovasz_192_224_on_pseudo_000005_cyclic_2_upd',
        # 'unet_resnext_50_lovasz_192_224_on_pseudo_000005_cyclic_3_upd',
        # 'unet_resnext_50_lovasz_192_224_on_pseudo_000005'

    ]

    model_pathes = [args.models_dir + x for x in models]


    phalanx_dict4 = read_phalanx_test('res34_resize128_newpseudo_v2.npy')

    phalanx_dicts = [phalanx_dict4]

    pred, predicted_masks, predicted_probs = ensemble(model_pathes, [0, 1, 2, 3, 4],
                                                      test.id.values, 0.5,
                                                      classification='',
                                                      phalanx_dicts=phalanx_dicts,
                                                      weights=[],
                                                      inner_weights=[0, 1])
    MODEL_PRED = pred

    test = pd.read_csv(os.path.join(args.data_root, 'sample_submission.csv'))
    test['rle_mask'] = MODEL_PRED
    test[['id', 'rle_mask']].to_csv('13_88735.csv', index=False)