import torch


def calculate_iou(preds, labels, smooth=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    labels = labels.float()

    dims = (1, 2, 3) if preds.dim() == 4 else (1, 2)

    intersection = (preds * labels).sum(dims)
    union = (preds + labels).sum(dims) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def calculate_dice(preds, labels, smooth=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    labels = labels.float()

    dims = (1, 2, 3) if preds.dim() == 4 else (1, 2)

    intersection = (preds * labels).sum(dims)
    dice = (2. * intersection + smooth) / (preds.sum(dims) + labels.sum(dims) + smooth)

    return dice.mean().item()


def get_metrics(preds, labels):
    return {
        'iou': calculate_iou(preds, labels),
        'dice': calculate_dice(preds, labels)
    }
