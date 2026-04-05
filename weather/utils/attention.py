import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_target_layer(model):
    if hasattr(model, 'model'):
        model = model.model
    if hasattr(model, 'base_model'):
        model = model.base_model

    layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if layers:
        return [layers[-1]]

    return None


def show_attention(model, model_name, img_tensor, label_idx, class_names, device):
    model.to(device)
    model.eval()

    is_vit = "vit" in model_name.lower()
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    mean, std = (np.array([0.5] * 3), np.array([0.5] * 3)) if is_vit else (np.array([0.485, 0.456, 0.406]),
                                                                           np.array([0.229, 0.224, 0.225]))
    img_rgb = np.clip(std * img_np + mean, 0, 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"Original: {class_names[label_idx]}")
    plt.axis('off')

    if is_vit:
        actual_model = model.model if hasattr(model, 'model') else model

        with torch.no_grad():
            outputs = actual_model(img_tensor.unsqueeze(0).to(device), output_attentions=True)

        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            print(f"{model_name} не вернул attention weights!")
            return

        attentions_tuple = outputs.attentions

        if len(attentions_tuple) == 0:
            print("Нет слоёвa внимания для отображения.")
            return

        last_layer_att = attentions_tuple[-1]

        att_mat = torch.mean(last_layer_att, dim=1).squeeze(0)

        v = att_mat[0, 1:]

        grid_size = int(np.sqrt(v.size(-1)))

        mask = v.reshape(grid_size, grid_size).detach().cpu().numpy()
        mask = cv2.resize(mask / (mask.max() + 1e-8), (img_rgb.shape[1], img_rgb.shape[0]))

        cam_image = show_cam_on_image(img_rgb, mask, use_rgb=True)

    elif "mlp" in model_name.lower() or "perceptron" in model_name.lower():
        print(f"Модель {model_name} (MLP) не имеет пространственного внимания.")
        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, "MLP has no\nspatial attention", ha='center')
        plt.axis('off')
        plt.show()
        return

    else:
        target_layers = get_target_layer(model)
        if target_layers:
            cam = GradCAM(model=model, target_layers=target_layers)
            targets = [ClassifierOutputTarget(label_idx)]
            grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0).to(device), targets=targets)[0, :]
            cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
        else:
            print(f"Не удалось найти целевой слой для {model_name}")
            return

    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title(f"Attention: {model_name}")
    plt.axis('off')
    plt.show()
