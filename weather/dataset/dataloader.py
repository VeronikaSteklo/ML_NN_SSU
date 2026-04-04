import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split

from .transforms import get_train_transforms, get_val_transforms
from .dataset import ApplyTransform


def get_weather_loaders(data_path, batch_size=64, image_size=128, seed=42):
    full_dataset = datasets.ImageFolder(root=data_path)
    indices = np.arange(len(full_dataset))
    labels = full_dataset.targets

    temp_idx, test_idx = train_test_split(
        indices, test_size=0.15, stratify=labels, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        temp_idx, test_size=0.2, stratify=[labels[i] for i in temp_idx], random_state=seed
    )

    train_tr = get_train_transforms(image_size)
    val_tr = get_val_transforms(image_size)

    train_data = ApplyTransform(Subset(full_dataset, train_idx), transform=train_tr)
    val_data = ApplyTransform(Subset(full_dataset, val_idx), transform=val_tr)
    test_data = ApplyTransform(Subset(full_dataset, test_idx), transform=val_tr)

    data = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    loaders = {
        'train': DataLoader(train_data, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_data, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_data, batch_size=batch_size, shuffle=False)
    }

    return data, loaders, full_dataset.targets, full_dataset.classes
