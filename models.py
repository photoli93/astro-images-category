import torch
import torch.nn as nn
from torchvision import models

class SpaceClassifier(nn.Module):
    """EfficientNet-B0 based classifier for space images"""
    
    def __init__(self, num_classes, pretrained=True):
        super(SpaceClassifier, self).__init__()
        
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Freeze backbone during stage 1
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    # Unfreeze only the last block of the EfficientNet backbone
    def unfreeze_backbone_partial(self):
        for name, param in self.backbone.named_parameters():
            if "features.7" in name or "classifier" in name:
                param.requires_grad = True
        print("Only final block + classifier unfrozen")

    # Gradual unfreeze function
    def unfreeze_backbone_stage(self, stage=2):
        for name, param in self.backbone.named_parameters():
            if stage == 1:
                if "features.7" in name or "classifier" in name:
                    param.requires_grad = True
            elif stage == 2:
                if any(f"features.{i}" in name for i in [6, 7]) or "classifier" in name:
                    param.requires_grad = True
            elif stage == 3:
                param.requires_grad = True
        print(f"Backbone unfrozen for stage {stage}")

