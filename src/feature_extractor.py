from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models


def _load_backbone(name: str) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Return backbone and its preprocessing transforms."""
    name = name.lower()
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        backbone = models.mobilenet_v3_small(weights=weights)
        backbone.classifier[-1] = torch.nn.Identity()
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        backbone = models.mobilenet_v3_large(weights=weights)
        backbone.classifier[-1] = torch.nn.Identity()
    elif name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        backbone.fc = torch.nn.Identity()
    else:
        raise ValueError(
            f"Unknown encoder '{name}'. "
            "Use one of: mobilenet_v3_small, mobilenet_v3_large, resnet18."
        )
    return backbone, weights.transforms()


class FeatureExtractor:
    """Projects a person crop into an embedding space used for similarity."""

    def __init__(self, device: str = "cpu", encoder: str = "mobilenet_v3_small") -> None:
        self.device = torch.device(device)
        backbone, transforms_pipeline = _load_backbone(encoder)
        self.backbone = backbone.to(self.device).eval()
        self.transform = transforms_pipeline

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.backbone(tensor)
            embedding = torch.nn.functional.normalize(embedding, dim=1)
        return embedding.squeeze(0).cpu().numpy()

    @staticmethod
    def preprocess_box(
        box: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))
        return x1, y1, x2, y2
