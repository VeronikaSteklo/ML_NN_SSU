import albumentations as A
from albumentations.pytorch import ToTensorV2
from carvana.config import IMAGE_SIZE, IMAGE_NORMALIZATION_MEAN, IMAGE_NORMALIZATION_STD
import cv2
import numpy as np


class AugmentBackground(A.DualTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    def apply(self, img, mask=None, **params):
        if mask is None:
            return img

        bg = cv2.GaussianBlur(img, (51, 51), 0)
        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)

        mask_3d = np.expand_dims(mask, axis=-1)

        return np.where(mask_3d > 0, img, bg)

    def apply_to_mask(self, mask, **params):
        return mask


def get_transforms(image_size=IMAGE_SIZE, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),

            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=15,
                p=0.5
            ),

            A.Normalize(
                mean=IMAGE_NORMALIZATION_MEAN,
                std=IMAGE_NORMALIZATION_STD,
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(
                mean=IMAGE_NORMALIZATION_MEAN,
                std=IMAGE_NORMALIZATION_STD,
            ),
            ToTensorV2(),
        ])


def get_hr_transforms(crop_size=512, is_train=True):
    if is_train:
        return A.Compose([
            A.RandomCrop(height=crop_size, width=crop_size),

            AugmentBackground(p=0.4),

            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),

            A.Normalize(mean=IMAGE_NORMALIZATION_MEAN, std=IMAGE_NORMALIZATION_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.CenterCrop(height=crop_size, width=crop_size),
            A.Normalize(mean=IMAGE_NORMALIZATION_MEAN, std=IMAGE_NORMALIZATION_STD),
            ToTensorV2(),
        ])

