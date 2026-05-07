"""
models/cnn_extractor.py

CNN backbone (ResNet18/50 or EfficientNet-B0) used for visual feature extraction.

The final classification layer is replaced with a projection head
that maps features into a fixed embedding space (embedding_dim).
"""

import torch
import torch.nn as nn
from torchvision import models
from config import CNN_BACKBONE, CNN_FREEZE_LAYERS


class CNNExtractor(nn.Module):
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()

        self.embedding_dim = embedding_dim
        weights = "IMAGENET1K_V1" if pretrained else None
 

        if CNN_BACKBONE == "resnet18":
            base = models.resnet18(weights=weights)
            in_features = base.fc.in_features 
            base.fc = nn.Identity()

        elif CNN_BACKBONE == "resnet50":
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features  
            base.fc = nn.Identity()

        elif CNN_BACKBONE == "efficientnet_b0":
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features  
            base.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {CNN_BACKBONE}")
 

        if CNN_FREEZE_LAYERS:
            for param in base.parameters():
                param.requires_grad = False
 
            if CNN_BACKBONE in ("resnet18", "resnet50"):
                for param in base.layer3.parameters():
                    param.requires_grad = True
                for param in base.layer4.parameters():
                    param.requires_grad = True

            elif CNN_BACKBONE == "efficientnet_b0":
                # Unfreeze last 3 blocks
                blocks = list(base.features.children())
                for block in blocks[-3:]:
                    for param in block.parameters():
                        param.requires_grad = True

        self.backbone = base
 

        self.projector = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [B, 3, 224, 224]

        Output:
            embedding: [B, embedding_dim]
        """
        feat = self.backbone(x) 
        feat = feat.view(feat.size(0), -1)
        emb = self.projector(feat) 
        return emb

    def trainable_params(self) -> int:
        """Returns number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)