import os

import torch
from PIL import Image
from torchvision import transforms


def generate_manual_aug(source_dir, save_dir, n_variants=3, image_size=128):
    diverse_transforms = [
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),

        transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),

        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]

    to_pil = transforms.ToPILImage()
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for class_name in sorted(os.listdir(source_dir)):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        target_class_path = os.path.join(save_dir, class_name)
        os.makedirs(target_class_path, exist_ok=True)

        print(f"Обрабатываю класс: {class_name}")

        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(valid_extensions)]

        for img_name in image_files:
            img_path = os.path.join(class_path, img_name)

            try:
                img = Image.open(img_path).convert('RGB')

                for i in range(min(n_variants, len(diverse_transforms))):
                    try:
                        transform = diverse_transforms[i] if i < len(diverse_transforms) else diverse_transforms[0]
                        aug_tensor = transform(img)

                        if aug_tensor.min() < 0 or aug_tensor.max() > 1:
                            aug_tensor = torch.clamp(aug_tensor, 0, 1)

                        aug_img = to_pil(aug_tensor)
                        aug_img.save(os.path.join(target_class_path, f"div_aug_{i}_{img_name}"))

                    except Exception as e:
                        print(f"  Ошибка аугментации {img_name} с трансформом {i}: {e}")

            except Exception as e:
                print(f"  Пропущен файл {img_name}: {e}")

    print("Генерация разнообразных аугментаций завершена!")
