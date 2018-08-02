import numpy as np
import keras
import cv2
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, ids, labels=None, is_train=True, batch_size=32, dim=128, n_channels=1,
                 n_classes=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.is_train = is_train
        self.batch_size = batch_size
        self.labels = labels
        self.ids = ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
#         if self.is_train:
#             return int(np.ceil(len(self.ids) / self.batch_size) * 2)
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        # If I'd like to increase number of steps?
#         if index not in range(0, len(self.file_list)):
#             return self.__getitem__(np.random.randint(0, self.__len__()))
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        ids_temp = [self.ids[k] for k in indexes]

        # Generate data
        X, y = self._data_generation(ids_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    @staticmethod
    def strong_aug(p=.5):
        return Compose([
            RandomRotate90(),
            Flip(),
            Transpose()
        ], p=p)        
            
    def _read_image_mask(self, id):
        img = cv2.imread('train/images/{}.png'.format(id), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.dim, self.dim))
    #resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
        mask = cv2.imread('train/masks/{}.png'.format(id), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.dim, self.dim))

        if self.is_train:
            augmentation = self.strong_aug(p=0.75)
            data = {"image": img, "mask": mask}
            augmented = augmentation(**data)
            img, mask = augmented["image"], augmented["mask"]

        img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)
        
        img = np.array(img, np.float32) / 255
        mask = np.array(mask, np.float32) / 255
        
        return (img,mask)
    
    def _data_generation(self, ids_temp):
        'Generates data containing batch_size samples'
        
        X = np.empty((self.batch_size, self.dim, self.dim, self.n_channels), np.float32)
        y = np.empty((self.batch_size, self.dim, self.dim, self.n_channels), np.float32)
        
        for idx, image_id in enumerate(ids_temp):
            X[idx,], y[idx,] = self._read_image_mask(image_id)

        return X, y
    