import os
import numpy as np
import cv2
from tqdm import tqdm_notebook, tqdm
from collections import defaultdict
import multiprocessing
from contextlib import contextmanager
from losses import iou_metric, jacard_coef_np, dice_coef_np
import threading
from params import args
from albumentations import PadIfNeeded, CenterCrop, HorizontalFlip, RandomContrast


class ThreadsafeIter(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def __next__(self):
        with self.lock:
            return next(self.it)


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def freeze_model(model, freeze_before_layer):
    if freeze_before_layer == "ALL":
        for l in model.layers:
            l.trainable = False
    else:
        freeze_before_layer_index = -1
        for i, l in enumerate(model.layers):
            if l.name == freeze_before_layer:
                freeze_before_layer_index = i
        for l in model.layers[:freeze_before_layer_index + 1]:
            l.trainable = False


def do_tta(img, TTA, preprocess):
    imgs = []

    if TTA == 'flip':
        augmentation = HorizontalFlip(p=1)
        data = {'image': img}
        img2 = augmentation(**data)['image']

        for im in [img, img2]:
            im = np.array(im, np.float32)

            im = cv2.resize(im, (args.resize_size, args.resize_size))
            augmentation = PadIfNeeded(min_height=args.input_size, min_width=args.input_size, p=1.0, border_mode=4)
            data = {"image": im}
            im = augmentation(**data)["image"]
            im = np.array(im, np.float32)

            imgs.append(preprocess(im))

    elif TTA == 'flip_brightness':
        augmentation = HorizontalFlip(p=1)
        data = {'image': img}
        img2 = augmentation(**data)['image']

        augmentation = RandomContrast(p=1, limit=0.2)
        data = {'image': img}
        img3 = augmentation(**data)['image']

        # img3 = np.roll(img, 10, axis=1)

        for im in [img, img2, img3]:
            im = np.array(im, np.float32)

            im = cv2.resize(im, (args.resize_size, args.resize_size))
            augmentation = PadIfNeeded(min_height=args.input_size, min_width=args.input_size, p=1.0, border_mode=4)
            data = {"image": im}
            im = augmentation(**data)["image"]
            im = np.array(im, np.float32)

            imgs.append(preprocess(im))

    return imgs


def undo_tta(imgs, TTA, preprocess):
    part = []
    for img in imgs:
        augmentation = CenterCrop(height=args.resize_size, width=args.resize_size, p=1.0)
        data = {"image": img}
        prob = augmentation(**data)["image"]
        # prob = cv2.resize(prob, (args.initial_size, args.initial_size), interpolation = cv2.INTER_AREA)
        prob = cv2.resize(prob, (args.initial_size, args.initial_size))

        part.append(prob)

    augmentation = HorizontalFlip(p=1)
    data = {'image': part[1]}
    part[1] = augmentation(**data)['image']

    # part[2] = np.roll(part[2], 91, axis=1)

    part = np.mean(np.array(part), axis=0)

    return part


def read_image_test(id, TTA, oof, preprocess):
    if oof:
        img = cv2.imread(os.path.join(args.images_dir, '{}.png'.format(id)), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(os.path.join(args.test_folder, '{}.png'.format(id)), cv2.IMREAD_COLOR)

    imgs = do_tta(img, TTA, preprocess)

    return imgs


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


def _get_augmentations_count(TTA=''):
    if TTA == '':
        return 1

    elif TTA == 'flip':
        return 2

    elif TTA == 'flip_brightness':
        return 3

    else:
        'No Such TTA'


def predict_test(model, preds_path, oof, ids, batch_size, thr=0.5, TTA='', preprocess=None):
    num_images = ids.shape[0]
    rles = []
    for start in tqdm(range(0, num_images, batch_size)):
        end = min(start + batch_size, num_images)

        augment_number_per_image = _get_augmentations_count(TTA)
        images = [read_image_test(x, TTA=TTA, oof=oof, preprocess=preprocess) for x in ids[start:end]]
        images = [item for sublist in images for item in sublist]

        X = np.array([x for x in images])
        preds = model.predict_on_batch(X)

        total = 0
        for idx in range(end - start):
            part = []

            part = undo_tta(preds[total:total + augment_number_per_image], TTA, preprocess)
            total += augment_number_per_image

            cv2.imwrite(os.path.join(preds_path, str(ids[start + idx]) + '.png'), np.array(part * 255, np.uint8))
            mask = part > thr
            rle = RLenc(mask)
            rles.append(rle)

    return rles


def postprocessing(final_pred):
    # morphological closing
    kernel = np.ones((3, 3), np.uint8)
    final_pred = cv2.morphologyEx(np.array(final_pred, dtype=np.uint8) * 255, cv2.MORPH_CLOSE, kernel) / 255

    #     #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    #     final_pred = cv2.morphologyEx(np.array(final_pred,dtype=np.uint8)*255, cv2.MORPH_CLOSE, kernel)/255

    #     kernel = np.ones((5,3),np.uint8)
    #     final_pred = cv2.morphologyEx(np.array(final_pred,dtype=np.uint8)*255, cv2.MORPH_CLOSE, kernel)/255

    # Connected components:
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        np.array(final_pred, dtype=np.uint8) * 255, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 5

    # your answer image
    final_pred = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            final_pred[output == i + 1] = 1

    # Rectangles
    #     if (final_pred.sum() > 0) and np.all(np.sum(np.abs(final_pred-final_pred[-1])) < 10) and np.any(final_pred[-1]!=1):
    #         final_pred = np.array([list(final_pred[-1]),]*101)
    #         #final_pred = np.array([list(np.mean(final_pred,-1)),]*101)

    num_of_pixel_in_mask = final_pred.sum()
    if num_of_pixel_in_mask <= 5:
        final_pred = np.zeros(final_pred.shape)

    return final_pred


def ensemble(model_dirs, folds, ids, thr, classification, phalanx_dicts=None, weights=None, inner_weights=None):
    rles = []
    predicted_masks = {}
    predicted_probs = {}

    if weights is None:
        weights = [1] * len(model_dirs)

    for img_id in tqdm(ids):
        preds = []
        for d, w in zip(model_dirs, weights):
            pred_folds = []
            prob_all_folds = 1
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
            # final_pred = np.mean(np.array(preds), axis=0)
            final_pred = np.sum(np.array(preds), axis=0) / sum(weights)
        else:
            # final_pred = np.mean(np.array(preds), axis=0)

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

        # mask = postprocessing(mask)

        rle = RLenc(mask)
        rles.append(rle)

        predicted_probs[img_id] = final_pred
        predicted_masks[img_id] = mask * 255

    return rles, predicted_masks, predicted_probs


def evaluate(model_dirs, ids, thr, classification, snapshots, leaks_dict=None, weights=None):
    metrics = defaultdict(list)

    if weights is None:
        weights = [1] * len(model_dirs)

    for img_id in tqdm(ids):
        preds = []
        for d, w in zip(model_dirs, weights):
            for snap in snapshots:
                path = os.path.join(d, snap)
                mask = cv2.imread(os.path.join(path, '{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)

                img = cv2.imread(os.path.join(args.images_dir, '{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)
                if np.unique(img).shape[0] == 1:
                    preds.append(np.zeros(mask.shape))
                else:
                    preds.append(np.array(mask / 255, np.float32) * w)

        final_pred = np.sum(np.array(preds), axis=0) / sum(weights)
        final_pred = final_pred > thr
        # final_pred = postprocessing(final_pred)

        if leaks_dict:
            if img_id in leaks_dict.keys():
                final_pred = leaks_dict[img_id]

        true_mask = cv2.imread(os.path.join(args.masks_dir, '{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)
        true_mask = np.array(true_mask / 255, np.float32)

        metrics['iout'].append(iou_metric(true_mask, final_pred))
        metrics['dice'].append(dice_coef_np(true_mask, final_pred))
        metrics['jacard'].append(jacard_coef_np(true_mask, final_pred))

    return metrics


def generate_pseudolabels(model_dirs, folds, ids, thr, phalanx_dict=None, weights=None):
    pseudolabels = {}

    if weights is None:
        weights = [1] * len(model_dirs)

    for img_id in tqdm(ids):
        preds = []
        for d, w in zip(model_dirs, weights):
            pred_folds = []
            prob_all_folds = 1
            for fold in folds:
                path = os.path.join(d, 'fold_{}'.format(fold))
                mask = cv2.imread(os.path.join(path, '{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)

                img = cv2.imread(os.path.join(args.test_folder, '{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)
                if np.unique(img).shape[0] == 1:
                    pred_folds.append(np.zeros(mask.shape))
                else:
                    pred_folds.append(np.array(mask / 255, np.float32))

            preds.append(np.mean(np.array(pred_folds, np.float32), axis=0) * w)

        if phalanx_dict is None:
            # final_pred = np.mean(np.array(preds), axis=0)
            final_pred = np.sum(np.array(preds), axis=0) / sum(weights)
        else:
            # final_pred = [np.mean(np.array(preds), axis=0)]
            final_pred = [np.sum(np.array(preds), axis=0) / sum(weights)]
            final_pred.append(phalanx_dict[img_id])
            final_pred = np.mean(np.array(final_pred), axis=0)
        confidence = (np.sum(final_pred < 0.2) + np.sum(final_pred > 0.8)) / (101 ** 2)

        mask = final_pred > thr

        pseudolabels[img_id] = (confidence, mask * 255)

    return pseudolabels

