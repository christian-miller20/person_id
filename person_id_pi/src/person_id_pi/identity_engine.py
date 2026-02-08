from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .face_types import FaceEmbedding, IdentityDecision, TrackletEmbedding
from .identity_config import IdentityConfig
from .identity_store import IdentityStore


@dataclass
class MatchResult:
    user_id: str
    score: float


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Embedding shape mismatch")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _median_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    stacked = np.stack(embeddings, axis=0)
    return np.median(stacked, axis=0).astype(np.float32)


def _dispersion(embeddings: List[np.ndarray], reference: np.ndarray) -> float:
    if not embeddings:
        return 0.0
    scores = [_cosine(reference, emb) for emb in embeddings]
    if len(scores) == 1:
        return 0.0
    return float(np.std(scores))


class IdentityEngine:
    def __init__(self, store: IdentityStore, config: Optional[IdentityConfig] = None) -> None:
        self.store = store
        self.config = config or IdentityConfig()

    def aggregate_tracklet(self, embeddings: Iterable[FaceEmbedding]) -> TrackletEmbedding:
        filtered = [
            _normalize(item.embedding)
            for item in embeddings
            if item.quality >= self.config.quality_min
        ]
        if not filtered:
            return TrackletEmbedding(
                embedding=np.zeros((512,), dtype=np.float32),
                n_used=0,
                dispersion=0.0,
                outliers=0,
            )
        median = _normalize(_median_embedding(filtered))
        inliers: List[np.ndarray] = []
        outliers = 0
        for emb in filtered:
            if _cosine(median, emb) >= self.config.outlier_threshold:
                inliers.append(emb)
            else:
                outliers += 1
        if not inliers:
            inliers = [median]
        mean = _normalize(np.mean(np.stack(inliers, axis=0), axis=0))
        dispersion = _dispersion(inliers, mean)
        return TrackletEmbedding(
            embedding=mean,
            n_used=len(inliers),
            dispersion=dispersion,
            outliers=outliers,
        )

    def match(self, tracklet: TrackletEmbedding) -> IdentityDecision:
        if tracklet.n_used < self.config.n_min:
            return IdentityDecision(
                user_id=None,
                score=0.0,
                margin=0.0,
                accepted=False,
                reason="insufficient_samples",
            )
        if tracklet.dispersion > self.config.dispersion_max:
            return IdentityDecision(
                user_id=None,
                score=0.0,
                margin=0.0,
                accepted=False,
                reason="high_dispersion",
            )

        best: Optional[MatchResult] = None
        second: Optional[MatchResult] = None
        for user_id in self.store.list_users():
            templates = self.store.get_templates(user_id)
            if not templates:
                continue
            score = max(_cosine(tracklet.embedding, _normalize(t)) for t in templates)
            result = MatchResult(user_id=user_id, score=score)
            if best is None or score > best.score:
                second = best
                best = result
            elif second is None or score > second.score:
                second = result

        if best is None:
            return IdentityDecision(
                user_id=None,
                score=0.0,
                margin=0.0,
                accepted=False,
                reason="no_templates",
            )

        margin = best.score - (second.score if second else 0.0)
        accepted = (
            best.score >= self.config.accept_threshold
            and margin >= self.config.margin_threshold
        )
        reason = "accepted" if accepted else "below_threshold"
        return IdentityDecision(
            user_id=best.user_id if accepted else None,
            score=best.score,
            margin=margin,
            accepted=accepted,
            reason=reason,
        )

    def update_templates(self, user_id: str, tracklet: TrackletEmbedding) -> bool:
        if tracklet.n_used < self.config.n_min:
            return False
        existing = self.store.get_templates(user_id)
        if existing:
            best = max(_cosine(tracklet.embedding, _normalize(t)) for t in existing)
            if best >= self.config.add_threshold:
                return False
        updated = existing + [tracklet.embedding]
        if len(updated) > self.config.max_templates_per_user:
            updated = self._prune_templates(updated, self.config.max_templates_per_user)
        self.store.replace_templates(user_id, updated)
        return True

    def _prune_templates(self, templates: List[np.ndarray], max_count: int) -> List[np.ndarray]:
        if len(templates) <= max_count:
            return templates
        kept: List[np.ndarray] = [templates[0]]
        while len(kept) < max_count:
            candidate_scores: List[Tuple[float, np.ndarray]] = []
            for template in templates:
                if any(np.allclose(template, k) for k in kept):
                    continue
                min_sim = min(_cosine(template, k) for k in kept)
                candidate_scores.append((min_sim, template))
            if not candidate_scores:
                break
            candidate_scores.sort(key=lambda item: item[0])
            kept.append(candidate_scores[0][1])
        return kept
