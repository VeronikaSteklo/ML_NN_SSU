import random

import numpy as np
import torch

DATA_PATH = "../data/weather_dataset"
MY_DATA_TEST_PATH = "../data/test_weather"
AUG_DATA_TEST_PATH = "../data/aug_weather"
GEN_DATA_TEST_PATH = "../data/gen_weather"

MODELS_SAVE_PATH = "../models/weather"
IMAGE_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
SEED = 42


def set_seed(seed=42):
    """Устанавливает seed для воспроизводимости"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
