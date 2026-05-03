import torch, os, yaml, pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

from bccd.dataset.dataset import CountDataset
from bccd.models.models import ResNetCount
from bccd.models.metrics import regression_metrics


def train_regression(model_path="outputs/models/regression_resnet.pth", val_ratio=0.2, seed=42):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    classes = cfg["CLASSES"]

    df = pd.read_csv("dataset/regression_targets.csv")
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed, stratify=None)

    train_transform = transforms.Compose([
        transforms.Resize((cfg["IMG_SIZE"], cfg["IMG_SIZE"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((cfg["IMG_SIZE"], cfg["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = CountDataset(train_df, train_transform)
    val_ds = CountDataset(val_df, val_transform)
    train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    model = ResNetCount(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["LR"]))
    criterion = torch.nn.MSELoss()

    Path("outputs/models").mkdir(parents=True, exist_ok=True)

    print(f"Training on {device} | Train: {len(train_df)}, Val: {len(val_df)}")

    patience = 10
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(int(cfg["NUM_EPOCHS"])):
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, y_true, y_pred = 0, [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, targets).item()
                y_true.append(targets.cpu().numpy())
                y_pred.append(torch.round(preds).clamp(min=0).cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        metrics = regression_metrics(y_true, y_pred, classes)

        print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_loss / len(val_loader):.4f} | "
              f"Global MAE: {metrics['Global']['MAE']:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return model_path
