import timm
from torchvision import models
from torch import nn


def get_mobilenet(num_classes):
    return timm.create_model('mobilenetv3_small_100.lamb_in1k', pretrained=True, num_classes=num_classes)


def get_resnet50(num_classes):
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_efficientnet_b2(num_classes):
    return timm.create_model('efficientnet_b2', pretrained=True, num_classes=num_classes)
