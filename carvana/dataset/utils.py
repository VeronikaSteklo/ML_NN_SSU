import os
from PIL import Image


def get_image_info(image_path):
    """Получить информацию об изображении"""
    try:
        img = Image.open(image_path)
        return {
            'size': img.size,
            'mode': img.mode,
            'format': img.format
        }
    except Exception as e:
        return {'error': str(e)}


def check_dataset_integrity(images_dir, masks_dir):
    """Проверить целостность датасета"""
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.gif')]

    missing_masks = []
    missing_images = []

    for img_file in image_files:
        mask_file = img_file.replace('.jpg', '_mask.gif')
        if mask_file not in mask_files:
            missing_masks.append(img_file)

    for mask_file in mask_files:
        img_file = mask_file.replace('_mask.gif', '.jpg')
        if img_file not in image_files:
            missing_images.append(mask_file)

    return {
        'total_images': len(image_files),
        'total_masks': len(mask_files),
        'missing_masks': missing_masks,
        'missing_images': missing_images
    }


def calculate_mask_statistics(dataset, num_samples=100):
    """Рассчитать статистику масок"""
    if len(dataset) == 0:
        return {}

    num_samples = min(num_samples, len(dataset))

    foreground_pixels = 0
    total_pixels = 0

    for i in range(num_samples):
        _, mask = dataset[i]
        mask_np = mask.squeeze().numpy()
        foreground_pixels += (mask_np > 0.5).sum()
        total_pixels += mask_np.size

    foreground_ratio = foreground_pixels / total_pixels if total_pixels > 0 else 0

    return {
        'foreground_pixels': foreground_pixels,
        'total_pixels': total_pixels,
        'foreground_ratio': foreground_ratio,
        'background_ratio': 1 - foreground_ratio
    }
