import cv2
import numpy as np
from params import args
import os
from albumentations import RandomCrop, CenterCrop, PadIfNeeded


class SegmentationDataGenerator:
    def __init__(self, input_shape=(128,128), batch_size=32, preprocess=None, augs=None):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.augs = augs
    
    @staticmethod 
    def get_down_masks(y1, new_img_size=128):
        y2 = [resize(m, (new_img_size//2, new_img_size//2), preserve_range=True, order=3) for m in (y1) ]
        y3 = [resize(m, (new_img_size//4, new_img_size//4), preserve_range=True, order=3) for m in (y1) ]
        y4 = [resize(m, (new_img_size//8, new_img_size//8), preserve_range=True, order=3) for m in (y1) ]
        y5 = [resize(m, (new_img_size//16, new_img_size//16), preserve_range=True, order=3) for m in (y1) ]
        y2 = [(m > 0.5).astype('float32') for m in y2]
        y3 = [(m > 0.5).astype('float32') for m in y3]
        y4 = [(m > 0.5).astype('float32') for m in y4]
        y5 = [(m > 0.5).astype('float32') for m in y5]
        y2 = np.array(y2).reshape(-1, new_img_size//2, new_img_size//2, 1).astype('float32')
        y3 = np.array(y3).reshape(-1, new_img_size//4, new_img_size//4, 1).astype('float32')
        y4 = np.array(y4).reshape(-1, new_img_size//8, new_img_size//8, 1).astype('float32')
        y5 = np.array(y5).reshape(-1, new_img_size//16, new_img_size//16, 1).astype('float32')

        y6 = np.amax(y1, axis=(1,2), keepdims=True)

        return [y6, y1, y2, y3, y4, y5, y1]    
    
    def _read_image_train(self, id):
        # Read with resizing

        
        if os.path.isfile(os.path.join(args.images_dir,'{}.png'.format(id))):
            img = cv2.imread(os.path.join(args.images_dir,'{}.png'.format(id)), cv2.IMREAD_COLOR)
            mask = cv2.imread(os.path.join(args.masks_dir,'{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
            
        else:
            img = cv2.imread(os.path.join(args.test_folder,'{}.png'.format(id)), cv2.IMREAD_COLOR)
            mask = cv2.imread(os.path.join(args.pseudolabels_dir,'{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
            

        img = np.array(img, np.float32)

        if self.augs:
            data = {"image": img, "mask": mask}
            augmented = self.augs(**data)
            img, mask = augmented["image"], augmented["mask"]

        img = cv2.resize(img, (args.resize_size,args.resize_size))
        mask = cv2.resize(mask, (args.resize_size,args.resize_size))
        augmentation = PadIfNeeded(min_height=self.input_shape[0], min_width=self.input_shape[1], p=1.0, border_mode=4) 
        data = {"image": img, "mask": mask}
        augmented = augmentation(**data)
        img, mask = augmented["image"], augmented["mask"]

        img = np.array(img, np.float32)
        img = self.preprocess(img)

        mask = np.array(mask / 255., np.float32)
        #mask = np.where(mask < 0.5, 0, 1)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, axis=2)
        return (img, mask)

    def _read_image_valid(self, id):
        img = cv2.imread(os.path.join(args.images_dir,'{}.png'.format(id)), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(args.masks_dir,'{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)

        img = np.array(img, np.float32)

        img = cv2.resize(img, (args.resize_size,args.resize_size))
        mask = cv2.resize(mask, (args.resize_size,args.resize_size))
        augmentation = PadIfNeeded(min_height=self.input_shape[0], min_width=self.input_shape[1], p=1.0, border_mode=4) 
        data = {"image": img, "mask": mask}
        augmented = augmentation(**data)
        img, mask = augmented["image"], augmented["mask"]

        img = np.array(img, np.float32)
        img = self.preprocess(img)

        mask = np.array(mask / 255., np.float32)
        #mask = np.where(mask < 0.5, 0, 1)
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

                
class ClassificationDataGenerator:
    def __init__(self, input_shape=(101,101), batch_size=32, preprocess=None, augs=None):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.augs = augs
        
    def _read_image_train(self, id):
        img = cv2.imread(os.path.join(args.images_dir,'{}.png'.format(id)), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(args.masks_dir,'{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
        
        y = 1 if np.max(mask) == 255 else 0
        
        img = np.array(img, np.float32)

        if self.augs:
            data = {"image": img}
            augmented = self.augs(**data)
            img = augmented["image"]
        
        img = np.array(img, np.float32)
        img = self.preprocess(img)
        
        return (img, y)


    def _read_image_valid(self, id):
        img = cv2.imread(os.path.join(args.images_dir,'{}.png'.format(id)), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(args.masks_dir,'{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
        
        y = 1 if np.max(mask) == 255 else 0
        
        img = np.array(img, np.float32)
        img = self.preprocess(img)
        
        return (img, y)
    

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