import torch
from ..config import MODELS_SAVE_PATH, DEVICE
from ..models.CNN import CNN
from ..models.perceptron import Perceptron
from ..models.vit import get_vit_model
from ..models.transfer_models import get_resnet50, get_mobilenet, get_efficientnet_b2


def load_all_weather_models(class_names):
    num_classes = len(class_names)
    loaded_models = {}

    constructors = {
        "MLP": lambda: Perceptron(128 * 128 * 3, num_classes),
        "CNN": lambda: CNN(num_classes),
        "ResNet50": lambda: get_resnet50(num_classes),
        "MobileNet": lambda: get_mobilenet(num_classes),
        "EfficientNet_B2": lambda: get_efficientnet_b2(num_classes),
        "ViT": lambda: get_vit_model(class_names, DEVICE, output_attentions=True)
    }

    for name in constructors:
        print(f"Загружаю {name}...")
        if name == "ViT":
            model, info = constructors[name]()
        else:
            model = constructors[name]()
        path = f"{MODELS_SAVE_PATH}/final_{name.lower()}_weather.pth"

        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            if name != "ViT":
                loaded_models[name] = model
            else:
                loaded_models[name] = {"model": model, "info": info}
            print(f" Успешно: {name} готова.")
        except FileNotFoundError:
            print(f" Ошибка: Файл {path} не найден.")

    return loaded_models
