import torch.nn as nn
import torchvision.models as models


class ResNetCount(nn.Module):
    def __init__(self, num_classes=3, backbone="resnet50", pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_features, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        features = self.backbone(x)
        # Счёт не может быть отрицательным
        counts = self.relu(self.head(features))
        return counts
