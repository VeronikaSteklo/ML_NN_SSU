import albumentations as A
from albumentations.pytorch import ToTensorV2
from carvana.config import IMAGE_SIZE, IMAGE_NORMALIZATION_MEAN, IMAGE_NORMALIZATION_STD


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
