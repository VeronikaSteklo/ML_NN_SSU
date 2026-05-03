import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd

cfg = yaml.safe_load(open('configs/default.yaml'))
CLASSES = cfg['CLASSES']


class CountDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = ["WBC", "RBC", "Platelets"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        targets = torch.tensor([row[c] for c in self.classes], dtype=torch.float32)
        return img, targets


class BloodCountDataset(Dataset):
    def __init__(self, csv_path, train=True):
        self.df = pd.read_csv(csv_path)
        self.transform = transforms.Compose([
            transforms.Resize((cfg['IMG_SIZE'], cfg['IMG_SIZE'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) if train else transforms.Compose([
            transforms.Resize((cfg['IMG_SIZE'], cfg['IMG_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.transform(Image.open(row['path']).convert('RGB'))
        targets = torch.tensor([row[c] for c in CLASSES], dtype=torch.float32)
        return img, targets
