import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from carvana.train.metrics import get_metrics

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"[EarlyStopping] Счетчик: {self.counter} из {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.counter = 0


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    summary = {'loss': 0, 'iou': 0, 'dice': 0}

    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()

        metrics = get_metrics(outputs, masks)

        summary['loss'] += loss.item()
        summary['iou'] += metrics['iou']
        summary['dice'] += metrics['dice']

        pbar.set_postfix(loss=loss.item(), iou=metrics['iou'])

    for key in summary: summary[key] /= len(loader)
    return summary


@torch.no_grad()
def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    summary = {'loss': 0, 'iou': 0, 'dice': 0}

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)

        loss = loss_fn(outputs, masks)
        metrics = get_metrics(outputs, masks)

        summary['loss'] += loss.item()
        summary['iou'] += metrics['iou']
        summary['dice'] += metrics['dice']

    for key in summary: summary[key] /= len(loader)
    return summary


def run_training(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, save_dir='outputs', patience=3, min_delta=0.01, model_name=None):
    os.makedirs(save_dir, exist_ok=True)
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}
    best_iou = 0.0

    loss_name = loss_fn.__class__.__name__
    if model_name is not None:
        loss_name = f"{model_name}_{loss_name}"
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    for epoch in range(epochs):
        train_res = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_res = validate_epoch(model, val_loader, loss_fn, device)

        if val_res['iou'] > best_iou:
            best_iou = val_res['iou']
            model_path = f"{save_dir}/best_model_{loss_name}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"[*] Лучшая модель сохранена (IoU: {best_iou:.4f})")

        for key in history:
            if 'train' in key:
                history[key].append(train_res[key.replace('train_', '')])
            else:
                history[key].append(val_res[key.replace('val_', '')])

        early_stopping(val_res['loss'])
        if early_stopping.early_stop:
            print(f"Ранняя остановка на {epoch + 1:02} эпохе")
            break

        print(
            f"Epoch {epoch + 1:02d} | Val Loss: {val_res['loss']:.4f} | IoU: {val_res['iou']:.4f} | Dice: {val_res['dice']:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_title('Loss History');
    ax1.legend()

    ax2.plot(history['val_iou'], label='IoU')
    ax2.plot(history['val_dice'], label='Dice')
    ax2.set_title('Metrics History');
    ax2.legend()

    plt.savefig(f"{save_dir}/training_plots_{loss_name}.png")
    plt.show()

    return history
