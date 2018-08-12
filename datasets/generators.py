import cv2
import numpy as np
from utils import read_image, read_mask
from params import args
import os

class DataGenerator:
    def __init__(self, input_shape=(128,128), batch_size=32, preprocess=None, augs=None):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.augs = augs
        
    def _read_image_train(self, id):
        img = read_image(os.path.join(args.images_dir,'{}.png'.format(id)), self.input_shape)
        mask = read_mask(os.path.join(args.masks_dir,'{}.png'.format(id)), self.input_shape)
        
        if self.augs:
            data = {"image": img, "mask": mask}
            augmented = self.augs(**data)
            img, mask = augmented["image"], augmented["mask"]
            
        img = np.array(img, np.float32)
        img = self.preprocess(img)
        
        mask = np.array(mask / 255., np.float32)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, axis=2)
        
        return (img, mask)


    def _read_image_valid(self, id):
        img = read_image(os.path.join(args.images_dir,'{}.png'.format(id)), self.input_shape)
        mask = read_mask(os.path.join(args.masks_dir,'{}.png'.format(id)), self.input_shape)
        
        img = np.array(img, np.float32)
        img = self.preprocess(img)
        
        mask = np.array(mask / 255., np.float32)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, axis=2)

        return (img, mask)


    def train_batch_generator(self, ids):
        num_images = ids.shape[0]
        while True:
            idx_batch = np.random.randint(low=0, high=num_images, size=self.batch_size)

            image_masks = [self._read_image_train(x) for x in ids[idx_batch]]

            X = np.array([x[0] for x in image_masks])
            y = np.array([x[1] for x in image_masks])

            yield X, y


    def evaluation_batch_generator(self, ids):
        num_images = ids.shape[0]
        while True:
            for start in range(0, num_images, self.batch_size):
                end = min(start + self.batch_size, num_images)

                image_masks = [self._read_image_valid(x) for x in ids[start:end]]

                X = np.array([x[0] for x in image_masks])
                y = np.array([x[1] for x in image_masks])

                yield X, y

                
