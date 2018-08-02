import numpy as np
import pandas as pd
import os
import cv2
from params import args
from preprocessing import preprocess_img

class SaltDataset:
    def __init__(self,
                 images_dir,
                 masks_dir,
                 fold=0,
                 n_folds=5,
                 seed=13,
                 ):
        self.fold = fold
        self.n_folds = n_folds
        self.seed = seed
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        np.random.seed(seed)
        self.train_ids, self.val_ids = self.generate_ids()
        print("Found {} train images".format(len(self.train_ids)))
        print("Found {} val images".format(len(self.val_ids)))
    
    def train_generator(self, input_shape=(128,128), batch_size=32, preprocessing_function=None, augs=None):
        num_images = self.train_ids.shape[0]
        while True:
            idx_batch = np.random.randint(low=0, high=num_images, size=batch_size)
            batch_x = []
            batch_y = []
            
            for img_id in self.train_ids[idx_batch]:
                img = cv2.imread(os.path.join(self.images_dir,'{}.png'.format(img_id)), cv2.IMREAD_COLOR)
                img = cv2.resize(img, input_shape)
                img = np.array(img, np.float32)
                img = preprocess_img(img, preprocessing_function)

                mask = cv2.imread(os.path.join(self.masks_dir,'{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, input_shape)
                mask = np.expand_dims(mask, axis=2)
                mask = np.array(mask / 255., np.float32)
                
                if augs:
                    data = {"image": img, "mask": mask}
                    augmented = augs(**data)
                    img, mask = augmented["image"], augmented["mask"]
                    if len(mask.shape) < 3:
                        mask = np.expand_dims(mask, axis=2)
                
                batch_x.append(img)
                batch_y.append(mask)
                
            batch_x = np.array(batch_x, dtype="float32")
            batch_y = np.array(batch_y, dtype="float32")
            
            yield batch_x, batch_y
        

    def val_generator(self, input_shape=(128,128), batch_size=64, preprocessing_function=None):
        num_images = self.val_ids.shape[0]
        while True:
            for start in range(0, num_images, self.batch_size):
                end = min(start + self.batch_size, num_images)
                
                batch_x = []
                batch_y = []
            
                for img_id in self.val_ids[start:end]:
                    img = cv2.imread(os.path.join(self.images_dir,'{}.png'.format(img_id)), cv2.IMREAD_COLOR)
                    img = cv2.resize(img, input_shape)
                    img = np.array(img, np.float32)
                    img = preprocess_img(img, preprocessing_function)

                    mask = cv2.imread(os.path.join(self.masks_dir,'{}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, input_shape)
                    mask = np.expand_dims(mask, axis=2)
                    mask = np.array(mask / 255., np.float32)

                    batch_x.append(img)
                    batch_y.append(mask)

            batch_x = np.array(batch_x, dtype="float32")
            batch_y = np.array(batch_y, dtype="float32")
            
            yield batch_x, batch_y
        

    def generate_ids(self):
        df = pd.read_csv(args.folds_csv)
        ids_train = df[(df.unique_pixels > 1) & (df.fold != self.fold)].id.values
        ids_valid = df[df.fold == self.fold].id.values

        return ids_train, ids_valid