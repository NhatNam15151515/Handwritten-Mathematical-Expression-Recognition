"""Patch-2 ViT classifier head for single-symbol recognition."""
import torch
import torch.nn as nn

from model import SharedPatchViTEncoder


class Patch2SymbolClassifier(nn.Module):
    def __init__(self, config, num_classes: int):
        super().__init__()
        self.vision_encoder = SharedPatchViTEncoder(config)
        self.classifier_norm = nn.LayerNorm(config.EMBED_DIM)
        self.classifier = nn.Linear(config.EMBED_DIM, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder.forward_features(images)
        pooled = features.mean(dim=1)
        return self.classifier(self.classifier_norm(pooled))
