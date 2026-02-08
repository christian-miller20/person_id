from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    bbox: Tuple[int, int, int, int]
    landmarks: Optional[List[Tuple[float, float]]]
    quality: float
    aligned_crop: np.ndarray


@dataclass(frozen=True)
class FaceEmbedding:
    embedding: np.ndarray  # shape (512,)
    quality: float
    bbox: Tuple[int, int, int, int]
    landmarks: Optional[List[Tuple[float, float]]]


@dataclass(frozen=True)
class Tracklet:
    embeddings: List[FaceEmbedding]

    def qualities(self) -> Iterable[float]:
        return (item.quality for item in self.embeddings)


@dataclass(frozen=True)
class TrackletEmbedding:
    embedding: np.ndarray
    n_used: int
    dispersion: float
    outliers: int


@dataclass(frozen=True)
class IdentityDecision:
    user_id: Optional[str]
    score: float
    margin: float
    accepted: bool
    reason: str
