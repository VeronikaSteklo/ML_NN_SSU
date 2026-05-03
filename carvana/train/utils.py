import cv2
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np


def get_best_metrics_table(histories):
    data = []
    for name, hist in histories.items():
        best_idx = np.argmax(hist['val_iou'])

        data.append({
            'Loss Function': name,
            'Best Val IoU': hist['val_iou'][best_idx],
            'Best Val Dice': hist['val_dice'][best_idx],
            'Loss at Best IoU': hist['val_loss'][best_idx],
            'Epochs Trained': len(hist['val_loss']),
            'Best Epoch': best_idx + 1
        })

    df = pd.DataFrame(data)

    df = df.sort_values(by='Best Val IoU', ascending=False).reset_index()

    return df

def visualize_model_comparison(models_dict, dataset, device, num_samples=3):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    num_models = len(models_dict)

    fig, axes = plt.subplots(num_samples, num_models + 2, figsize=(20, num_samples * 4))

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)

        img_display = image.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)

        axes[row, 0].imshow(img_display)
        axes[row, 0].set_title("Original Image")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis('off')

        for col, (name, model) in enumerate(models_dict.items()):
            model.eval()
            with torch.no_grad():
                pred = model(input_tensor)
                pred = torch.sigmoid(pred).squeeze().cpu().numpy()
                pred_binary = (pred > 0.5).astype(np.uint8)

            axes[row, col + 2].imshow(pred_binary, cmap='gray')
            axes[row, col + 2].set_title(f"Pred: {name}")
            axes[row, col + 2].axis('off')

    plt.tight_layout()
    plt.show()


def overlay_mask(img, mask, color=(0, 1, 1), alpha=0.6, intensity=1.5):
    mask_rgb = np.zeros_like(img)
    for i in range(3):
        mask_rgb[..., i] = mask * color[i] * intensity

    result = cv2.addWeighted(img, 1.0, mask_rgb, alpha, 0)
    return np.clip(result, 0, 1)


def show_results(models_dict, dataset, device, num_samples=3):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, len(models_dict) + 1, figsize=(20, num_samples * 5))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (std * img_np + mean).clip(0, 1)

        gt_mask = mask.squeeze().cpu().numpy()
        gt_overlay = overlay_mask(img_np, gt_mask, color=(0, 1, 0), alpha=0.5, intensity=2.0)
        axes[row, 0].imshow(gt_overlay)
        axes[row, 0].set_title("GT (Neon Green)")
        axes[row, 0].axis('off')

        for col, (name, model) in enumerate(models_dict.items()):
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(image.unsqueeze(0).to(device))).squeeze().cpu().numpy()
                pred_mask = (pred > 0.5).astype(np.float32)

            res_overlay = overlay_mask(img_np, pred_mask, color=(1, 1, 0), alpha=0.6, intensity=1.8)
            axes[row, col + 1].imshow(res_overlay)
            axes[row, col + 1].set_title(f"Pred: {name}")
            axes[row, col + 1].axis('off')

    plt.tight_layout()
    plt.show()