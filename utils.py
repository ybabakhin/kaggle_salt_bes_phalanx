import os
import pandas as pd
import cv2
from tqdm import tqdm_notebook
import logging
from collections import defaultdict

logger = logging.getLogger('log')

from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
import keras

import multiprocessing
from functools import partial
from contextlib import contextmanager

from losses import *
from models import models_zoo, unets
from callbacks import *
from augmentations import *

import threading
from params import args



from albumentations import PadIfNeeded, CenterCrop

def read_image(path, input_size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    augmentation = PadIfNeeded(min_height=input_size[0], min_width=input_size[1], p=1.0)
    data = {"image": img}
    img = augmentation(**data)["image"]
    
    # img = cv2.resize(img, input_size)
    
    return img
    
def read_mask(path, input_size):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    augmentation = PadIfNeeded(min_height=input_size[0], min_width=input_size[1], p=1.0)
    data = {"image": mask}
    mask = augmentation(**data)["image"]
    
    # mask = cv2.resize(mask, input_size)
    
    return mask  

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

    
def get_model(weights_path, model_name):
    if model_name == 'unet_128':
        input_size = (128,128,1)

        model = models_zoo.get_unet_128(input_size)

        model.compile(optimizer=RMSprop(lr=0.0001), loss=make_loss('bce_dice'),
                      metrics=[dice_coef, jacard_coef])

        augs = get_augmentations('initial',p=0.9)

        callbacks = get_callback('reduce_lr', weights_path=weights_path, early_stop_patience=20, reduce_lr_factor=0.5, reduce_lr_patience=10,
                                 reduce_lr_min=0.000001)


    elif model_name == 'unet_128_dropout_adam':
        input_size = (128,128,1)

        model = models_zoo.get_unet_128_do_adam(input_size)

        model.compile(optimizer=Adam(lr=0.0001), loss=make_loss('bce_dice'), metrics=[dice_coef, jacard_coef])

        augs = get_augmentations('initial',p=0.75)

        callbacks = get_callback('reduce_lr', weights_path=weights_path, early_stop_patience=10, reduce_lr_factor=0.25, reduce_lr_patience=5,
                                 reduce_lr_min=0.000001)

    elif model_name == 'unet_128_v2':
        input_size = (128,128,1)

        model = models_zoo.get_unet_128(input_size)

        model.compile(optimizer=RMSprop(lr=0.0001), loss=make_loss('bce_dice'),
                      metrics=[dice_coef, jacard_coef])

        augs = get_augmentations('initial',p=0.75)

        callbacks = get_callback('reduce_lr', weights_path=weights_path, early_stop_patience=10, reduce_lr_factor=0.25, reduce_lr_patience=5,
                                 reduce_lr_min=0.000001)
        
    elif model_name == 'unet_128_jacard':
        input_size = (128,128,3)

        model = models_zoo.get_unet_128(input_size)

        model.compile(optimizer=RMSprop(lr=0.0001), loss=make_loss('bce_jacard'),
                      metrics=[dice_coef, jacard_coef])

        augs = get_augmentations('initial',p=0.9)

        callbacks = get_callback('early_stopping', weights_path=weights_path, early_stop_patience=5)
        
        def preprocess(img):
            return img / 255.
        
    elif model_name == 'resnet_50_224':
        input_size = (224,224,3)

        model = models_zoo.get_unet_resnet_50(input_size)

        model.compile(optimizer=RMSprop(lr=0.0001), loss=make_loss('bce_jacard'),
                      metrics=[dice_coef, jacard_coef])

        augs = get_augmentations('initial',p=0.9)

        #callbacks = get_callback('early_stopping', weights_path=weights_path, early_stop_patience=5)
        callbacks = get_callback('reduce_lr', weights_path=weights_path, early_stop_patience=10, reduce_lr_factor=0.25, reduce_lr_patience=5,
                                 reduce_lr_min=0.000001)
        
        from keras.applications.resnet50 import preprocess_input
        
        def preprocess(img):
            return preprocess_input(img)
            
    elif model_name == 'resnet_50_224_old':
        input_size = (224,224,3)

        model = models_zoo.unet_resnet_50(input_size)

        model.compile(optimizer=RMSprop(lr=0.0001), loss=make_loss('bce_jacard'),
                      metrics=[dice_coef, jacard_coef])

        augs = get_augmentations('initial',p=0.9)

        #callbacks = get_callback('early_stopping', weights_path=weights_path, early_stop_patience=5)
        callbacks = get_callback('reduce_lr', weights_path=weights_path, early_stop_patience=10, reduce_lr_factor=0.25, reduce_lr_patience=5,
                                 reduce_lr_min=0.000001)
        
        from keras.applications.resnet50 import preprocess_input
        
        def preprocess(img):
            return preprocess_input(img)    
        
    elif model_name == 'resnet50_fpn_old':
        input_size = (128,128,3)

        model = unets.resnet50_fpn(input_size, channels=1, activation="sigmoid")

        model.compile(optimizer=RMSprop(lr=0.0001), loss=make_loss('bce_jacard'),
                      metrics=[dice_coef, jacard_coef])

        augs = get_augmentations('initial',p=0.9)

        #callbacks = get_callback('early_stopping', weights_path=weights_path, early_stop_patience=5)
        callbacks = get_callback('reduce_lr', weights_path=weights_path, early_stop_patience=10, reduce_lr_factor=0.25, reduce_lr_patience=5,
                                 reduce_lr_min=0.000001)
        
        from keras.applications.resnet50 import preprocess_input
        
        def preprocess(img):
            return preprocess_input(img)
        
    elif model_name == 'resnet50_fpn':
        
        model = unets.resnet50_fpn((args.input_size, args.input_size, 3), channels=1, activation="sigmoid")

        model.compile(optimizer=RMSprop(lr=args.learning_rate), loss=make_loss(args.loss_function),
                      metrics=[dice_coef, jacard_coef])

        augs = get_augmentations(args.augmentation_name, p=args.augmentation_prob)

        #callbacks = get_callback('early_stopping', weights_path=weights_path, early_stop_patience=5)
        callbacks = get_callback(args.callback, weights_path=weights_path, early_stop_patience=args.early_stop_patience, reduce_lr_factor=args.reduce_lr_factor, reduce_lr_patience=args.reduce_lr_patience, reduce_lr_min=args.reduce_lr_min)
        
        from keras.applications.resnet50 import preprocess_input
        
        def preprocess(img):
            return preprocess_input(img) 

    else:
        ValueError("Unknown Model")

    return model, callbacks, augs, preprocess


# Cumsum channel
# Create cumsum x
#     x_center_mean = x_img[border:-border, border:-border].mean()
#     x_csum = (np.float32(x_img)-x_center_mean).cumsum(axis=0)
#     x_csum -= x_csum[border:-border, border:-border].mean()
#     x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

# # Set some parameters
# im_width = 128
# im_height = 128
# border = 5
# im_chan = 2 # Number of channels: first is original and second cumsum(axis=0)
# n_features = 1 # Number of extra features, like depth
# path_train = '../input/train/'
# path_test = '../input/test/'


def noise_cv():
    DATA_ROOT = 'data/'
    n_fold = 5
    print(os.path.join(DATA_ROOT, 'depths.csv'))
    depths = pd.read_csv(os.path.join(DATA_ROOT, 'depths.csv'))
    depths.sort_values('z', inplace=True)
    depths.drop('z', axis=1, inplace=True)
    depths['fold'] = (list(range(n_fold)) * depths.shape[0])[:depths.shape[0]]
    depths.to_csv(os.path.join(DATA_ROOT, 'folds.csv'), index=False)


def read_image_test(id, TTA, oof, preprocess):
    if oof:
        img = read_image(os.path.join(args.images_dir,'{}.png'.format(id)), (args.input_size,args.input_size))
    else:
        img = read_image(os.path.join(args.test_folder,'{}.png'.format(id)), (args.input_size,args.input_size))
    
    imgs = []

    if TTA == '':
        img = np.array(img, np.float32)
        img = preprocess(img)
        imgs.append(img)

    elif TTA == 'flip':
        augmentation = HorizontalFlip(p=1)
        data = {'image': img}
        img2 = augmentation(**data)['image']
        
        for im in [img,img2]:
            im = np.array(im, np.float32)
            im = preprocess(im)
            imgs.append(im)

    elif TTA == 'D4':
        pass

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

    elif TTA == 'D4':
        return 8

    else:
        'No Such TTA'

        
# def do_tta(x, tta_type):
#     if tta_type == 'hflip':
#         # batch, img_col = 2
#         return flip_axis(x, 2)
#     else:
#         return x
# def undo_tta(pred, tta_type):
#     if tta_type == 'hflip':
#         # batch, img_col = 2
#         return flip_axis(pred, 2)
#     else:
#         return pred

def predict_test(model, preds_path, oof, ids, batch_size, thr=0.5, TTA='', preprocess=None):
    num_images = ids.shape[0]
    with poolcontext(processes=12) as pool:
        rles = []
        for start in tqdm_notebook(range(0, num_images, batch_size)):
            end = min(start + batch_size, num_images)

            augment_number_per_image = _get_augmentations_count(TTA)
            #images = pool.map(partial(read_image_test, TTA=TTA, oof=oof, preprocess = preprocess), ids[start:end])
            images = [read_image_test(x, TTA=TTA, oof=oof, preprocess = preprocess) for x in ids[start:end]]

            images = [item for sublist in images for item in sublist]

            X = np.array([x for x in images])

            preds = model.predict_on_batch(X)

            total = 0
            for idx in range(end - start):
                part = []
                for aug in range(augment_number_per_image):
                    
                    # prob = cv2.resize(preds[total], (args.initial_size,args.initial_size))
                    
                    augmentation = CenterCrop(height = args.initial_size, width = args.initial_size, p = 1.0)
                    data = {"image": preds[total]}
                    prob = augmentation(**data)["image"]
                    
                    if aug == 0:
                        pass
                    elif aug == 1:
                        augmentation = HorizontalFlip(p=1)
                        data = {'image': prob}
                        prob = augmentation(**data)['image']
                    part.append(prob)
                    total += 1
                part = np.mean(np.array(part), axis=0)
                cv2.imwrite(os.path.join(preds_path, str(ids[start + idx]) + '.png'), np.array(part * 255, np.uint8))
                mask = part > thr
                rle = RLenc(mask)
                rles.append(rle)

        return rles


def ensemble(model_dirs, folds, ids, thr):
    rles = []
    for img_id in tqdm_notebook(ids):
        preds = []
        for d in model_dirs:
            pred_folds = []
            for fold in folds:
                path = os.path.join(d, 'fold_{}'.format(fold))
                mask = cv2.imread(os.path.join(path, '{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)
                
                img = cv2.imread(os.path.join(args.test_folder, '{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)
                if np.unique(img).shape[0] == 1:
                    pred_folds.append(np.zeros(mask.shape))
                else:
                    pred_folds.append(np.array(mask / 255, np.float32))
            
            preds.append(np.mean(np.array(pred_folds), axis=0))
        final_pred = np.mean(np.array(preds), axis=0)
        # final_pred = cv2.blur(final_pred,(11,11))
        
        mask = final_pred > thr
        rle = RLenc(mask)
        rles.append(rle)
    return rles

import scipy
def evaluate(model_dirs, ids, thr):
    metrics = defaultdict(list)

    for img_id in tqdm_notebook(ids):
        preds = []
        for d in model_dirs:
            path = os.path.join(d, 'oof')
            mask = cv2.imread(os.path.join(path, '{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)
            
            img = cv2.imread(os.path.join(args.images_dir, '{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)
            if np.unique(img).shape[0] == 1:
                preds.append(np.zeros(mask.shape))
            else:
                preds.append(np.array(mask / 255, np.float32))
            
            
        
        # final_pred = scipy.stats.mstats.gmean(np.array(preds), axis=0)
        final_pred = np.mean(np.array(preds), axis=0)

        true_mask = cv2.imread(os.path.join(args.masks_dir, '{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)
        true_mask = np.array(true_mask / 255, np.float32)
        
        # https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
        # final_pred = cv2.blur(final_pred,(11,11))
        final_pred = final_pred > thr

        
        metrics['iout'].append(iou_metric(true_mask, final_pred))
        metrics['dice'].append(dice_coef_np(true_mask, final_pred))
        metrics['jacard'].append(jacard_coef_np(true_mask, final_pred))

    return metrics

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

# Для тех, кому лень придумывать визуализацию

# ```ims = []
# for image, ytrue, ypred in zip(images.cpu().data.numpy()[i::4],
#                                ytrues.cpu().data.numpy()[i::4],
#                                ypreds.cpu().data.numpy()[i::4]):
#     image = np.swapaxes((np.swapaxes(image, 1, 2) + 2.15) / 4.8, 0, 2)[:, :, ::-1]
#     ytrue = ytrue[0]
#     ypred = ypred[0]

#     yprob = np_sigmoid(ypred)
#     ypred = (yprob > 0.5).astype(np.float32)

#     yfp = ypred * (1 - ytrue)
#     yfn = (1 - ypred) * ytrue
#     ytp = ypred * ytrue
#     ytn = (1 - ypred) * (1 - ytrue)

#     yshow = np.zeros_like(image)
#     yshow[..., 2] = yfn
#     yshow[..., 1] = yfp
#     # yshow += (yfp + yfn)[..., np.newaxis] * (0.5 * image)
#     yshow += ytp[..., np.newaxis] * (0.9 + 0.1 * image)
#     yshow += ytn[..., np.newaxis] * (0.1 * image)

#     ydiff = 0.5 + 0.5 * (yprob - ytrue)

#     im = np.hstack([np.uint8(image * 255),
#                     np.uint8(yshow * 255),
#                     colorize(yprob),
#                     colorize(ydiff)])
#     ims.append(im)
# im = np.vstack(ims)
# if training:
#     cv2.imwrite(f'output/{name}/fold{fold}/train/{epoch + 1:03d}_{suffix}_{i + 1}.png', im)
# else:
#     cv2.imwrite(f'output/{name}/fold{fold}/valid/{i + 1}_{epoch + 1:03d}_{suffix}.png', im)```

