import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(self, images_dir, masks_dir, csv_file=None, transform=None,
                 image_size=None, normalize_mask=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_size = image_size
        self.normalize_mask = normalize_mask

        if csv_file and os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
            self.image_names = self.df['img'].unique().tolist()
        else:
            self.image_names = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        mask_name = img_name.replace('.jpg', '_mask.gif')
        mask_path = os.path.join(self.masks_dir, mask_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

        try:
            mask_pil = Image.open(mask_path)
            mask_array = np.array(mask_pil)
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            raise

        if self.normalize_mask:
            mask_array = (mask_array > 0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask_array)
            image = augmented['image']
            mask = augmented['mask']
            
            if isinstance(mask, torch.Tensor):
                if mask.dtype == torch.uint8:
                    mask = mask.float()
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
            else:
                mask = torch.from_numpy(mask).float()
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
        else:
            if self.image_size is not None:
                image_pil = Image.fromarray(image).resize(
                    (self.image_size[1], self.image_size[0]), Image.BILINEAR
                )
                image = np.array(image_pil)
                mask_pil = Image.fromarray(mask_array).resize(
                    (self.image_size[1], self.image_size[0]), Image.NEAREST
                )
                mask_array = np.array(mask_pil)
            
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask_array).unsqueeze(0).float()

        return image, mask


class CarvanaHRDataset(CarvanaDataset):
    def __init__(self, images_dir, masks_dir, **kwargs):
        kwargs['image_size'] = None
        super().__init__(images_dir, masks_dir, **kwargs)

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)

        return image, mask

def visualize_sample(dataset, idx=None, denormalize=True):
    if idx is None:
        idx = np.random.randint(0, len(dataset))

    image, mask = dataset[idx]

    if denormalize and image.dim() == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = image.clamp(0, 1)

    image_np = image.permute(1, 2, 0).numpy()
    
    if image_np.dtype == np.float32 or image_np.dtype == np.float64:
        image_np = (image_np * 255).astype(np.uint8)
    
    mask_np = mask.squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def dataset_info(images_dir, masks_dir, csv_file=None):
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.gif')]

    print(f"Количество изображений: {len(image_files)}")
    print(f"Количество масок: {len(mask_files)}")

    matched = 0
    sample_size = min(10, len(image_files))
    for img_file in image_files[:sample_size]:
        mask_file = img_file.replace('.jpg', '_mask.gif')
        if mask_file in mask_files:
            matched += 1

    print(f"Соответствующих пар (пример из {sample_size}): {matched}/{sample_size}")

    if image_files:
        try:
            sample_img = Image.open(os.path.join(images_dir, image_files[0]))
            print(f"Размер изображений: {sample_img.size}")
        except:
            print("Не удалось получить размер изображения")

    if csv_file and os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            print(f"Записей в CSV: {len(df)}")
        except:
            print("Не удалось прочитать CSV файл")

    return image_files, mask_files
