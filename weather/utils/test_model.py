import math
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from weather.dataset.transforms import get_val_transforms


def test_model(model, model_name, data_path, class_names, device, max_errors=30, info=None):
    size = 224 if "vit" in model_name.lower() else 128
    transform = get_val_transforms(size)

    try:
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

    model.to(device)
    model.eval()

    correct = 0
    errors = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)

            if preds == labels:
                correct += 1
            else:
                if len(errors) < max_errors:
                    errors.append({
                        'image': inputs.cpu().squeeze(),
                        'true': class_names[labels.item()],
                        'pred': class_names[preds.item()],
                        'conf': conf.item()
                    })

    accuracy = correct / len(dataset)
    print(f"Результат {model_name}: {accuracy:.2%} ({correct}/{len(dataset)})")

    if errors:
        num_errors = len(errors)
        cols = 10
        rows = math.ceil(num_errors / cols)

        plt.figure(figsize=(20, rows * 2.2))
        plt.suptitle(f"Ошибки {model_name} | Accuracy: {accuracy:.2%}", fontsize=16)

        for i, err in enumerate(errors):
            plt.subplot(rows, cols, i + 1)

            img = err['image'].permute(1, 2, 0).numpy()

            if info is not None:
                mean = info.get('mean')
                std = info.get('std')
            else:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])

            img = std * img + mean
            img = np.clip(img, 0, 1)

            plt.imshow(img)
            plt.title(f"T:{err['true'][:5]}\nP:{err['pred'][:5]}",
                      color='red', fontsize=8, pad=2)
            plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print(f"Ошибок нет! {model_name} отработала идеально.")

    return accuracy
