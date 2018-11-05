import os
import numpy as np
import cv2
from tqdm import tqdm
import threading
from params import args
from albumentations import PadIfNeeded, CenterCrop, HorizontalFlip


class ThreadsafeIter(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def __next__(self):
        with self.lock:
            return next(self.it)


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

    return imgs


def undo_tta(imgs, TTA):
    part = []
    for img in imgs:
        augmentation = CenterCrop(height=args.resize_size, width=args.resize_size, p=1.0)
        data = {"image": img}
        prob = augmentation(**data)["image"]
        prob = cv2.resize(prob, (args.initial_size, args.initial_size))

        part.append(prob)

    if TTA == 'flip':
        augmentation = HorizontalFlip(p=1)
        data = {'image': part[1]}
        part[1] = augmentation(**data)['image']

    part = np.mean(np.array(part), axis=0)

    return part


def read_image_test(id, TTA, preprocess):
    img = cv2.imread(os.path.join(args.test_folder, '{}.png'.format(id)), cv2.IMREAD_COLOR)
    imgs = do_tta(img, TTA, preprocess)

    return imgs


def _get_augmentations_count(TTA=''):
    if TTA == '':
        return 1

    elif TTA == 'flip':
        return 2

    else:
        raise ValueError('No Such TTA')


def predict_test(model, preds_path, ids, batch_size, TTA='', preprocess=None):
    num_images = ids.shape[0]

    for start in tqdm(range(0, num_images, batch_size)):
        end = min(start + batch_size, num_images)

        augment_number_per_image = _get_augmentations_count(TTA)
        images = [read_image_test(x, TTA=TTA, preprocess=preprocess) for x in ids[start:end]]
        images = [item for sublist in images for item in sublist]

        X = np.array([x for x in images])
        preds = model.predict_on_batch(X)

        total = 0
        for idx in range(end - start):
            part = undo_tta(preds[total:total + augment_number_per_image], TTA)
            total += augment_number_per_image

            cv2.imwrite(os.path.join(preds_path, str(ids[start + idx]) + '.png'), np.array(part * 255, np.uint8))

