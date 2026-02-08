from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    # (x1, y1, x2, y2) bounding box in image coordinates.
    bbox: Tuple[int, int, int, int]
    # Optional 2D facial landmarks (e.g., 5-point) in image coordinates.
    landmarks: Optional[List[Tuple[float, float]]]
    # Detector confidence score used as a lightweight "quality" signal.
    quality: float
    # Aligned or raw face crop used for embedding (BGR image array).
    aligned_crop: np.ndarray
    # Optional 512-d embedding attached by the detector (InsightFace provides this).
    embedding: Optional[np.ndarray] = None


@dataclass(frozen=True)
class FaceEmbedding:
    # L2-normalized 512-d face embedding vector.
    embedding: np.ndarray  # shape (512,)
    # Quality score inherited from the detector.
    quality: float
    # Bounding box of the face used for this embedding.
    bbox: Tuple[int, int, int, int]
    # Optional facial landmarks used to derive the crop/alignment.
    landmarks: Optional[List[Tuple[float, float]]]


@dataclass(frozen=True)
class Tracklet:
    # List of per-frame embeddings for a single person in a clip.
    embeddings: List[FaceEmbedding]

    def qualities(self) -> Iterable[float]:
        return (item.quality for item in self.embeddings)


@dataclass(frozen=True)
class TrackletEmbedding:
    # Aggregated embedding after filtering/outlier rejection.
    embedding: np.ndarray
    # Number of embeddings used to build the aggregate.
    n_used: int
    # Dispersion of inlier similarities (lower is more consistent).
    dispersion: float
    # Count of outlier embeddings rejected during aggregation.
    outliers: int


@dataclass(frozen=True)
class IdentityDecision:
    # Matched user ID if accepted, otherwise None.
    user_id: Optional[str]
    # Best cosine similarity score across all users.
    score: float
    # Margin between best and second-best scores.
    margin: float
    # Whether the decision met acceptance thresholds.
    accepted: bool
    # Human-readable reason for acceptance/rejection.
    reason: str
