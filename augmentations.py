from albumentations import (VerticalFlip,
                            HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
                            Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
                            IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
                            IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, ElasticTransform
                            )


def get_augmentations(augmentation, p=0.5):
    if augmentation == 'initial':
        augmentations = Compose([
                RandomRotate90(),
                Flip(),
                Transpose()
            ], p=p)
        
    elif augmentation == 'all':
        
        augmentations =  Compose([
                #a CLAHE(clip_limit=2, p=0.35),
                #c GaussNoise(),
                #ToGray(prob=0.25),
                #InvertImg(prob=0.2),
                #Remap(p=0.4),
                RandomRotate90(),
                Flip(),
                Transpose(),
                #c Blur(blur_limit=3, p=.4),
                #a RandomContrast(p=.2),
                #c RandomBrightness(p=.2),
                #b ElasticTransform(p=0.3),
                #Distort1(p=0.3),
                #Distort2(p=.1),
                #b ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.4, rotate_limit=45, p=.7),
                #a HueSaturationValue(),
                #ChannelShuffle(prob=.2),
                #FixMasks(1.)
        ], p=p)

    else:
        ValueError("Unknown Augmentations")


    return augmentations

