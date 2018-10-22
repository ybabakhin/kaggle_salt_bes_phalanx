from albumentations import (HorizontalFlip, ShiftScaleRotate, RandomRotate90, Transpose,
                            RandomContrast, RandomBrightness, Flip, Compose, RandomCrop)


def get_augmentations(augmentation, p, input_shape):
    if augmentation == 'initial':
        augmentations = Compose([
            RandomRotate90(),
            Flip(),
            Transpose()
        ], p=p)

    elif augmentation == 'initial_crops':
        augmentations = Compose([
            RandomRotate90(0.9),
            Flip(0.9),
            Transpose(0.9),
            RandomCrop(height=input_shape[0], width=input_shape[1])
        ], p=p)

    elif augmentation == 'valid':
        augmentations = Compose([
            HorizontalFlip(p=.5),
            RandomBrightness(p=.2, limit=0.2),
            RandomContrast(p=.1, limit=0.2),
            ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7)
        ], p=p)

    elif augmentation == 'horizontal_flip':
        augmentations = Compose([
            HorizontalFlip(p=.5)
        ], p=p)

    else:
        ValueError("Unknown Augmentations")

    return augmentations
