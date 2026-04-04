from torchvision import transforms


def get_train_transforms(image_size=128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),

        # Геометрические аугментации
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),

        # Обрезка и изменение размера
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),

        # Цветовые аугментации
        transforms.ColorJitter(
            brightness=0.3,  # Изменение яркости ±30%
            contrast=0.3,  # Изменение контраста ±30%
            saturation=0.3,  # Изменение насыщенности ±30%
            hue=0.1  # Изменение оттенка ±10%
        ),

        # Дополнительные цветовые эффекты
        transforms.RandomGrayscale(p=0.1),

        # Шум и прочие эффекты
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),

        # Нормализация
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(image_size=128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
