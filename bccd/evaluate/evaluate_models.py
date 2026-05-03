import pandas as pd
import torch
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

from bccd.models.models import ResNetCount
from bccd.models.metrics import regression_metrics


def evaluate_all():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    classes = cfg["CLASSES"]

    df = pd.read_csv("dataset/regression_targets.csv")
    _, val_df = train_test_split(df, test_size=0.2, random_state=cfg["SEED"])

    y_true = val_df[classes].values

    print("Оценка регрессионной модели (ResNet50)")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    resnet = ResNetCount(num_classes=len(classes)).to(device)

    model_path = Path("outputs/models/regression_resnet.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Модель регрессии не найдена: {model_path}. Сначала запустите обучение.")

    resnet.load_state_dict(torch.load(model_path, map_location=device))
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((cfg["IMG_SIZE"], cfg["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    preds_resnet = []
    with torch.no_grad():
        for path in val_df["image_path"]:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            pred = torch.round(resnet(tensor)).cpu().numpy().squeeze()
            preds_resnet.append(pred)
    preds_resnet = np.array(preds_resnet)

    print("Оценка детекционной модели (YOLO)")
    yolo_path = Path("../runs/detect/outputs/detection_yolo/weights/best.pt")
    if not yolo_path.exists():
        raise FileNotFoundError(f"Модель YOLO не найдена: {yolo_path}. Сначала запустите обучение YOLO.")

    yolo = YOLO(str(yolo_path))
    preds_yolo = []

    for path in val_df["image_path"]:
        results = yolo.predict(path, imgsz=cfg["IMG_SIZE"], conf=0.25, verbose=False)
        boxes = results[0].boxes

        counts = {0: 0, 1: 0, 2: 0}  # 0: WBC, 1: RBC, 2: Platelets
        for cls_id in boxes.cls.cpu().numpy():
            counts[int(cls_id)] += 1

        preds_yolo.append([counts[0], counts[1], counts[2]])
    preds_yolo = np.array(preds_yolo)

    metrics_resnet = regression_metrics(y_true, preds_resnet, classes)
    metrics_yolo = regression_metrics(y_true, preds_yolo, classes)

    def print_metrics(name, m_dict):
        print(f"--- {name} ---")
        for cls_name in classes + ["Global"]:
            mae = m_dict[cls_name]['MAE']
            rmse = m_dict[cls_name]['RMSE']
            print(f"{cls_name:10} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
        print()

    print_metrics("Прямая регрессия (ResNet50)", metrics_resnet)
    print_metrics("Подсчет по детекции (YOLO)", metrics_yolo)


if __name__ == "__main__":
    evaluate_all()
