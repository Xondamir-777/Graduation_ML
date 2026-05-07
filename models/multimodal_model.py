"""
models/multimodal_model.py

Architecture:

  ┌────────────────────┐     ┌────────────────────────┐
  │ Image [3,224,224]  │     │ Climate features [7]   │
  └─────────┬──────────┘     └───────────┬────────────┘
            │ CNN (ResNet18)             │ Random Forest
            ↓ embedding [256]            ↓ probabilities [8]
            └──────────┬──────────────────┘
                       │ Concatenation [264]
                       ↓
                Fusion MLP Head
               [256 → 128 → 8]
                       ↓
            Disease class logits [8]
            + disease probability [1] (auxiliary head)
"""

import torch
import torch.nn as nn

from config import NUM_DISEASE_CLASSES
from models.cnn_extractor import CNNExtractor


# ─────────────────────────────────────────────────────────────
# Fusion Head
# ─────────────────────────────────────────────────────────────

class FusionHead(nn.Module):
    """
    MLP that fuses CNN embeddings and RF outputs.
    """

    def __init__(self, cnn_dim: int, rf_dim: int, num_classes: int):
        super().__init__()

        in_dim = cnn_dim + rf_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes),
        )

        self.prob_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, cnn_emb: torch.Tensor,
                rf_proba: torch.Tensor) -> dict:

        x = torch.cat([cnn_emb, rf_proba], dim=1)  # [B, cnn_dim + rf_dim]

        logits = self.classifier(x)                # [B, num_classes]
        prob = self.prob_head(x).squeeze(1)        # [B]

        return {
            "logits": logits,
            "prob": prob
        }


# ─────────────────────────────────────────────────────────────
# Full Multimodal Model
# ─────────────────────────────────────────────────────────────

class MultimodalDiseaseModel(nn.Module):
    """
    Full multimodal architecture:

    - CNN processes images → visual embedding
    - Random Forest provides climate-based probabilities
    - FusionHead combines both modalities
    """

    def __init__(
        self,
        cnn_embedding_dim: int = 256,
        rf_output_dim: int = NUM_DISEASE_CLASSES,
        num_classes: int = NUM_DISEASE_CLASSES,
        pretrained: bool = True
    ):
        super().__init__()

        self.cnn = CNNExtractor(
            embedding_dim=cnn_embedding_dim,
            pretrained=pretrained
        )

        self.fusion = FusionHead(
            cnn_dim=cnn_embedding_dim,
            rf_dim=rf_output_dim,
            num_classes=num_classes
        )

    def forward(self, image: torch.Tensor,
                rf_proba: torch.Tensor) -> dict:
        """
        Args:
            image:    [B, 3, 224, 224]
            rf_proba: [B, NUM_DISEASE_CLASSES] (from RF model)

        Returns:
            dict with:
                logits: classification output
                prob: auxiliary disease probability
        """
        cnn_emb = self.cnn(image)  # [B, 256]
        return self.fusion(cnn_emb, rf_proba)

    def get_cnn_embedding(self, image: torch.Tensor) -> torch.Tensor:
        """Returns CNN embeddings only (for analysis)."""
        return self.cnn(image)

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable
        }