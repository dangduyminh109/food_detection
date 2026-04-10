"""CNN model definitions for classification."""
import torch
import torch.nn as nn
from torchvision import models
from . import config

class FruitFreshnessModel(nn.Module):
    def __init__(self, model_name=config.MODEL_NAME, num_classes=config.NUM_CLASSES, 
                 pretrained=config.PRETRAINED, freeze_backbone=config.FREEZE_BACKBONE):
        super(FruitFreshnessModel, self).__init__()
        self.model_name = model_name.lower()
        
        if self.model_name == "resnet18":
            if hasattr(models, "ResNet18_Weights"):
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                self.model = models.resnet18(weights=weights)
            else:
                self.model = models.resnet18(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            
        elif self.model_name == "mobilenet_v2":
            if hasattr(models, "MobileNet_V2_Weights"):
                weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
                self.model = models.mobilenet_v2(weights=weights)
            else:
                self.model = models.mobilenet_v2(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            if self.model_name == "resnet18":
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            elif self.model_name == "mobilenet_v2":
                for param in self.model.classifier[1].parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.model(x)

def get_model():
    return FruitFreshnessModel()
