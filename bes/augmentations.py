from albumentations import (HorizontalFlip, ShiftScaleRotate, RandomContrast, RandomBrightness, Compose)


def get_augmentations(augmentation, p):
    if augmentation == 'valid':
        augmentations = Compose([
            HorizontalFlip(p=0.5),
            RandomBrightness(p=0.2, limit=0.2),
            RandomContrast(p=0.1, limit=0.2),
            ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7)
        ], p=p)

    else:
        ValueError("Unknown Augmentations")

    return augmentations
