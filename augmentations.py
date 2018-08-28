from albumentations import (VerticalFlip,
                            HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
                            Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
                            IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
                            IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, ElasticTransform, RandomCrop
                            )

def get_augmentations(augmentation, p, input_shape):
    if augmentation == 'initial':
        augmentations = Compose([
                RandomRotate90(),
                Flip(),
                Transpose()
            ], p=p)
        
    if augmentation == 'initial_crops':
        augmentations = Compose([
                RandomRotate90(0.9),
                Flip(0.9),
                Transpose(0.9),
                RandomCrop(height=input_shape[0], width=input_shape[1])
            ], p=p)
        
    elif augmentation == 'valid':
        augmentations = Compose([
                HorizontalFlip(p=.5),
                RandomBrightness(p=.2),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.4, rotate_limit=0, p=.7)
            ], p=p)
    # Distortion?    
    elif augmentation == 'valid_plus':
        augmentations = Compose([
            #RandomRotate90(p=1),
                HorizontalFlip(p=.5),
                RandomBrightness(p=.2,limit=0.2),
                RandomContrast(p=.1,limit=0.2),
                ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7),
            OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, p=0.2)
            ], p=p)


# OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, p=0.5)

# OneOf([
#             OpticalDistortion(p=0.3),
#             GridDistortion(p=0.1),
#             IAAPiecewiseAffine(p=0.3),
#         ], p=0.2),

# Try strong_aug

# TRY AGAIN:
# GaussNoise

# NO:
# VerticalFlip(p=.5),
# Transpose(0.5),
# Blur
        
    elif augmentation == 'horizontal_flip':
        augmentations = Compose([
                HorizontalFlip()
            ], p=p)
        
    elif augmentation == 'strong_aug':
        
        augmentations =  Compose([
        #RandomRotate90(),
        HorizontalFlip(p=0.5),
        #Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=0, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
    ], p=p)

    else:
        ValueError("Unknown Augmentations")


    return augmentations


