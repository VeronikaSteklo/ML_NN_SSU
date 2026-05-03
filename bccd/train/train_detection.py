import torch
from ultralytics import YOLO
import yaml


def train_yolo():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    cfg = yaml.safe_load(open("configs/default.yaml"))
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="dataset/yolo/data.yaml",
        epochs=cfg["NUM_EPOCHS"],
        imgsz=cfg["IMG_SIZE"],
        batch=cfg["BATCH_SIZE"],
        project="outputs",
        exist_ok=True,
        name="detection_yolo",
        save=True,
        device=device,
        patience=5,
        amp=False,
        workers=0,
    )

    metrics = results.results_dict
    return metrics
