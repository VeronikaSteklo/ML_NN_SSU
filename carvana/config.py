import os
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "carvana-image-masking-challenge"

TRAIN_IMAGES_DIR = DATA_DIR / "train"
TRAIN_MASKS_DIR = DATA_DIR / "train_masks"
TEST_IMAGES_DIR = DATA_DIR / "test"
METADATA_FILE = DATA_DIR / "metadata.csv"
TRAIN_MASKS_CSV = DATA_DIR / "train_masks.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"

IMAGE_SIZE = (256, 256)
ORIGINAL_IMAGE_SIZE = (1280, 1918)

IMAGE_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZATION_STD = [0.229, 0.224, 0.225]

AUGMENTATION_CONFIG = {
    'horizontal_flip_p': 0.5,  # Вероятность горизонтального отражения
    'rotate_limit': 15,  # Максимальный угол поворота (градусы)
    'rotate_p': 0.5,  # Вероятность поворота
    'shift_limit': 0.05,  # Лимит сдвига
    'scale_limit': 0.05,  # Лимит масштабирования
    'shift_scale_rotate_p': 0.5,  # Вероятность shift/scale/rotate
    'to_gray_p': 0.1,  # Вероятность конвертации в оттенки серого
}

BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

COLORS = {
    'background': [0, 0, 0],
    'car': [255, 255, 255]
}


def setup_paths():
    paths = [
        TRAIN_IMAGES_DIR,
        TRAIN_MASKS_DIR,
        TEST_IMAGES_DIR
    ]

    for path in paths:
        if not path.exists():
            print(f"Warning: Путь к файлу не найден: {path}")

    files = [
        METADATA_FILE,
        TRAIN_MASKS_CSV,
        SAMPLE_SUBMISSION
    ]

    for file in files:
        if not file.exists():
            print(f"Warning: Файл не существует: {file}")


setup_paths()
