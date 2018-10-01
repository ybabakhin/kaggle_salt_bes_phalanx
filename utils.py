import os
import pandas as pd
import cv2
from tqdm import tqdm_notebook, tqdm
from collections import defaultdict

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

import scipy

from albumentations import PadIfNeeded, CenterCrop, RandomCrop

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


def read_image_test(id, TTA, oof, preprocess):
    
    
    
    if oof:
        #img = read_image(os.path.join(args.images_dir,'{}.png'.format(id)), (args.input_size,args.input_size))
        img = cv2.imread(os.path.join(args.images_dir,'{}.png'.format(id)), cv2.IMREAD_COLOR)
    else:
        #img = read_image(os.path.join(args.test_folder,'{}.png'.format(id)), (args.input_size,args.input_size))
        img = cv2.imread(os.path.join(args.test_folder,'{}.png'.format(id)), cv2.IMREAD_COLOR)
    
    imgs = []

    if TTA == '':
        img = np.array(img, np.float32)
        
        img = cv2.resize(img, (args.resize_size,args.resize_size))
        augmentation = PadIfNeeded(min_height=args.input_size, min_width=args.input_size, p=1.0) 
        data = {"image": img}
        img = augmentation(**data)["image"]
        img = np.array(img, np.float32)
        
        imgs.append(preprocess(img))
        
        # Crops Prediction
#         imgs.append(preprocess(img[:args.input_size,:args.input_size,:].astype('float32')))
#         imgs.append(preprocess(img[-args.input_size:,-args.input_size:,:].astype('float32')))
#         imgs.append(preprocess(img[:args.input_size,-args.input_size:,:].astype('float32')))
#         imgs.append(preprocess(img[-args.input_size:,:args.input_size,:].astype('float32')))

    elif TTA == 'flip':
        augmentation = HorizontalFlip(p=1)
        data = {'image': img}
        img2 = augmentation(**data)['image']
        
        for im in [img,img2]:
            im = np.array(im, np.float32)
        
            im = cv2.resize(im, (args.resize_size,args.resize_size))
            augmentation = PadIfNeeded(min_height=args.input_size, min_width=args.input_size, p=1.0) 
            data = {"image": im}
            im = augmentation(**data)["image"]
            im = np.array(im, np.float32)

            imgs.append(preprocess(im))

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

def classification_predict_test(model, preds_path, oof, ids, batch_size, thr=0.5, TTA='', preprocess=None):
    num_images = ids.shape[0]
    probs = []
    for start in tqdm_notebook(range(0, num_images, batch_size)):
        end = min(start + batch_size, num_images)

        augment_number_per_image = _get_augmentations_count(TTA)
        images = [read_image_test(x, TTA=TTA, oof=oof, preprocess = preprocess) for x in ids[start:end]]
        images = [item for sublist in images for item in sublist]

        X = np.array([x for x in images])
        preds = model.predict_on_batch(X)

        total = 0
        for idx in range(end - start):
            part = []
            for aug in range(augment_number_per_image):

                prob = preds[total]

                if aug == 0:
                    pass
                elif aug == 1:
                    augmentation = HorizontalFlip(p=1)
                    data = {'image': prob}
                    prob = augmentation(**data)['image']
                part.append(prob)
                total += 1
            part = np.mean(np.array(part))
            probs.append(part)

    return probs

def predict_test(model, preds_path, oof, ids, batch_size, thr=0.5, TTA='', preprocess=None):
    num_images = ids.shape[0]
    rles = []
    for start in tqdm(range(0, num_images, batch_size)):
        end = min(start + batch_size, num_images)

        augment_number_per_image = _get_augmentations_count(TTA)
        images = [read_image_test(x, TTA=TTA, oof=oof, preprocess = preprocess) for x in ids[start:end]]
        images = [item for sublist in images for item in sublist]

        X = np.array([x for x in images])
        preds = model.predict_on_batch(X)

        total = 0
        for idx in range(end - start):
            part = []
            for aug in range(augment_number_per_image):

                # Crops prediction
#                     mask_zeroes = np.zeros((args.initial_size,args.initial_size,1))
#                     mask_zeroes_mult = np.zeros((args.initial_size,args.initial_size,1))

#                     mask_zeroes[:args.input_size,:args.input_size,:] += preds[total]
#                     mask_zeroes[-args.input_size:,-args.input_size:,:] += preds[total+1]
#                     mask_zeroes[:args.input_size,-args.input_size:,:] += preds[total+2]
#                     mask_zeroes[-args.input_size:,:args.input_size,:] += preds[total+3]

#                     mask_zeroes_mult[:args.input_size,:args.input_size,:] += 1
#                     mask_zeroes_mult[-args.input_size:,-args.input_size:,:] += 1
#                     mask_zeroes_mult[:args.input_size,-args.input_size:,:] += 1
#                     mask_zeroes_mult[-args.input_size:,:args.input_size,:] += 1

#                     prob = mask_zeroes/mask_zeroes_mult

                augmentation = CenterCrop(height = args.resize_size, width = args.resize_size, p = 1.0)
                data = {"image": preds[total]}
                prob = augmentation(**data)["image"]
                prob = cv2.resize(prob, (args.initial_size, args.initial_size))

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


def ensemble(model_dirs, folds, ids, thr, classification):
    rles = []
    if classification != '':
        df_dict = {'0':pd.read_csv(os.path.join(classification,'fold_0/probs_test_fold_0.csv')),
                  '1':pd.read_csv(os.path.join(classification,'fold_1/probs_test_fold_1.csv')),
                  '2':pd.read_csv(os.path.join(classification,'fold_2/probs_test_fold_2.csv')),
                  '3':pd.read_csv(os.path.join(classification,'fold_3/probs_test_fold_3.csv')),
                  '4':pd.read_csv(os.path.join(classification,'fold_4/probs_test_fold_4.csv'))}
                    
    for img_id in tqdm(ids):
        preds = []
        for d in model_dirs:
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
                
                if classification != '':
                    tt=df_dict[str(fold)]
                    prob = tt[tt.id==img_id].prob.values[0]
                    prob_all_folds*=prob  

            preds.append(np.mean(np.array(pred_folds), axis=0))
        final_pred = np.mean(np.array(preds), axis=0)
        # final_pred = cv2.blur(final_pred,(11,11))
     
        if classification != '':
            final_pred*=prob_all_folds**(1/5)
        
        mask = final_pred > thr
        
#         num_of_pixel_in_mask = mask.sum()
#         if num_of_pixel_in_mask <= 30:
#             mask = np.zeros(mask.shape)
        
        rle = RLenc(mask)
        rles.append(rle)
    return rles

def evaluate_exp(model_dirs, ids, thr, classification, low_value, pixels, del_pixels):
    metrics = defaultdict(list)

    for img_id in tqdm(ids):
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
        

        
        if classification != '':
            tt = pd.read_csv(os.path.join(classification,'probs_oof.csv'))
            prob = tt[tt.id==img_id].prob.values[0]
            if prob < 0.5:
                final_pred = np.zeros(mask.shape)
            #final_pred*=prob
        
        # https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
#         final_pred = cv2.blur(final_pred,(2,2))
        
        thr_mask_1 = np.zeros((101,101))
        thr_mask_1+=thr
        thr_mask_2 = np.zeros((101,101))
        thr_mask_2+=thr
        for row, const in enumerate(np.linspace(low_value, thr, pixels)):
            thr_mask_1[row, :pixels] = const
            thr_mask_1[row, -pixels:] = const
            thr_mask_1[-row, -pixels:] = const
            thr_mask_1[-row, :pixels] = const
            thr_mask_2[:pixels, row] = const
            thr_mask_2[-pixels:, row] = const
            thr_mask_2[-pixels:, -row] = const
            thr_mask_2[:pixels, -row] = const
        thr_mask = (thr_mask_1+thr_mask_2)/2
    
        final_pred = final_pred > thr_mask

        num_of_pixel_in_mask = final_pred.sum()
        if num_of_pixel_in_mask <= del_pixels:
            final_pred = np.zeros(mask.shape)
        
        metrics['iout'].append(iou_metric(true_mask, final_pred))
        metrics['dice'].append(dice_coef_np(true_mask, final_pred))
        metrics['jacard'].append(jacard_coef_np(true_mask, final_pred))

    return metrics

def evaluate(model_dirs, ids, thr, classification, snapshots):
    metrics = defaultdict(list)

    for img_id in tqdm(ids):
        preds = []
        for d in model_dirs:
            for snap in snapshots:
                path = os.path.join(d, snap)
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
        

        
        if classification != '':
            tt = pd.read_csv(os.path.join(classification,'probs_oof.csv'))
            prob = tt[tt.id==img_id].prob.values[0]
            if prob < 0.5:
                final_pred = np.zeros(mask.shape)
            #final_pred*=prob
        
        # https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
#         final_pred = cv2.blur(final_pred,(2,2))
        
        final_pred = final_pred > thr

#         num_of_pixel_in_mask = final_pred.sum()
#         if num_of_pixel_in_mask <= 30:
#             final_pred = np.zeros(mask.shape)
        
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

