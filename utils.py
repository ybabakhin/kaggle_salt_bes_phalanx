import os
import pandas as pd
import cv2
from tqdm import tqdm_notebook
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

# def read_image(path, input_size):
#     img = cv2.imread(path, cv2.IMREAD_COLOR)
#     img = cv2.resize(img, input_size)
#     # [:,:,::-1]

#     return img
    
# def read_mask(path, input_size):
#     mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     mask = cv2.resize(mask, input_size)

#     return mask  

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
                    
    #FROG
                    augmentation = CenterCrop(height = 128, width = 128, p = 1.0)
                    data = {"image": preds[total]}
                    prob = augmentation(**data)["image"]
                    prob = cv2.resize(prob, (args.initial_size,args.initial_size))
    
                    # prob = cv2.resize(preds[total], (args.initial_size,args.initial_size))
                    
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
                    
    for img_id in tqdm_notebook(ids):
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
        rle = RLenc(mask)
        rles.append(rle)
    return rles

def evaluate(model_dirs, ids, thr, classification):
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
        
        if classification != '':
            tt = pd.read_csv(os.path.join(classification,'probs_oof.csv'))
            prob = tt[tt.id==img_id].prob.values[0]
            if prob < 0.5:
                final_pred = np.zeros(mask.shape)
            #final_pred*=prob
        
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



# bad_masks =[
# '1eaf42beee.png'
# ,'33887a0ae7.png'
# ,'33dfce3a76.png'
# ,'3975043a11.png'
# ,'39cd06da7d.png'
# ,'483b35d589.png'
# ,'49336bb17b.png'
# ,'4ef0559016.png'
# ,'4fbda008c7.png'
# ,'4fdc882e4b.png'
# ,'50d3073821.png'
# ,'53e17edd83.png'
# ,'5b217529e7.png'
# ,'5f98029612.png'
# ,'608567ed23.png'
# ,'62aad7556c.png'
# ,'62d30854d7.png'
# ,'6460ce2df7.png'
# ,'6bc4c91c27.png'
# ,'7845115d01.png'
# ,'7deaf30c4a.png'
# ,'80a458a2b6.png'
# ,'81fa3d59b8.png'
# ,'8367b54eac.png'
# ,'849881c690.png'
# ,'876e6423e6.png'
# ,'90720e8172.png'
# ,'916aff36ae.png'
# ,'919bc0e2ba.png'
# ,'a266a2a9df.png'
# ,'a6625b8937.png'
# ,'a9ee40cf0d.png'
# ,'aeba5383e4.png'
# ,'b63b23fdc9.png'
# ,'baac3469ae.png'
# ,'be7014887d.png'
# ,'be90ab3e56.png'
# ,'bfa7ee102e.png'
# ,'bfbb9b9149.png'
# ,'c387a012fc.png'
# ,'c98dfd50ba.png'
# ,'caccd6708f.png'
# ,'cb4f7abe67.png'
# ,'d0bbe4fd97.png'
# ,'d4d2ed6bd2.png'
# ,'de7202d286.png'
# ,'f0c401b64b.png'
# ,'f19b7d20bb.png'
# ,'f641699848.png'
# ,'f75842e215.png'
# ,'00950d1627.png'
# ,'0280deb8ae.png'
# ,'06d21d76c4.png'
# ,'09152018c4.png'
# ,'09b9330300.png'
# ,'0b45bde756.png'
# ,'130229ec15.png'
# ,'15d76f1672.png'
# ,'182bfc6862.png'
# ,'23afbccfb5.png'
# ,'24522ec665.png'
# ,'285f4b2e82.png'
# ,'2bc179b78c.png'
# ,'2f746f8726.png'
# ,'3cb59a4fdc.png'
# ,'403cb8f4b3.png'
# ,'4f5df40ab2.png'
# ,'50b3aef4c4.png'
# ,'52667992f8.png'
# ,'52ac7bb4c1.png'
# ,'56f4bcc716.png'
# ,'58de316918.png'
# ,'640ceb328a.png'
# ,'71f7425387.png'
# ,'7c0b76979f.png'
# ,'7f0825a2f0.png'
# ,'834861f1b6.png'
# ,'87afd4b1ca.png'
# ,'88a5c49514.png'
# ,'9067effd34.png'
# ,'93a1541218.png'
# ,'95f6e2b2d1.png'
# ,'96216dae3b.png'
# ,'96523f824a.png'
# ,'99ee31b5bc.png'
# ,'9a4b15919d.png'
# ,'9b29ca561d.png'
# ,'9eb4a10b98.png'
# ,'ad2fa649f7.png'
# ,'b1be1fa682.png'
# ,'b24d3673e1.png'
# ,'b35b1b412b.png'
# ,'b525824dfc.png'
# ,'b7b83447c4.png'
# ,'b8a9602e21.png'
# ,'ba1287cb48.png'
# ,'be18a24c49.png'
# ,'c27409a765.png'
# ,'c2973c16f1.png'
# ,'c83d9529bd.png'
# ,'cef03959d8.png'
# ,'d4d34af4f7.png'
# ,'d9a52dc263.png'
# ,'dd6a04d456.png'
# ,'ddcb457a07.png'
# ,'e12cd094a6.png'
# ,'e6e3e58c43.png'
# ,'e73ed6e7f2.png'
# ,'f6e87c1458.png'
# ,'f7380099f6.png'
# ,'fb3392fee0.png'
# ,'fb47e8e74e.png'
# ,'febd1d2a67.png'
#     ]
# bad_masks = [x.split('.')[0] for x in bad_masks]