from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .face_types import FaceDetection, FaceEmbedding


@dataclass
class FaceEmbedderConfig:
    min_face_size: int = 80


class FaceEmbedder:
    """Stateless face detection + embedding interface.

    This is a stub to keep the identity pipeline testable before wiring
    in a concrete model (e.g., InsightFace or ArcFace).
    """

    def __init__(self, config: Optional[FaceEmbedderConfig] = None) -> None:
        self.config = config or FaceEmbedderConfig()

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        # TODO: integrate a face detector + landmark-based alignment.
        return []

    def embed(self, face: FaceDetection) -> FaceEmbedding:
        # TODO: integrate an ArcFace-style embedding model.
        embedding = np.zeros((512,), dtype=np.float32)
        return FaceEmbedding(
            embedding=embedding,
            quality=face.quality,
            bbox=face.bbox,
            landmarks=face.landmarks,
        )
