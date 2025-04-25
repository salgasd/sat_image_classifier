import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(
    width: int,
    height: int,
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
) -> albu.BaseCompose:
    transforms = []

    if preprocessing:
        transforms.append(albu.Resize(height=height, width=width))

    if augmentations:
        transforms.extend(
            [
                albu.RandomResizedCrop(height=height, width=width, p=1.0),
                albu.D4(p=1.0),
                albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                albu.GaussianBlur(),
                albu.CoarseDropout(),
            ],
        )

    if postprocessing:
        transforms.extend([albu.Normalize(), ToTensorV2()])

    return albu.Compose(transforms)
