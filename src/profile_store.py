from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Profile:
    profile_id: str
    embedding: List[float]
    num_samples: int = 1
    beers_consumed: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)
    history: List[List[float]] = field(default_factory=list)
    ema_embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "embedding": self.embedding,
            "num_samples": self.num_samples,
            "beers_consumed": self.beers_consumed,
            "metadata": self.metadata,
            "history": self.history,
            "ema_embedding": self.ema_embedding,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Profile":
        embedding = list(payload["embedding"])
        history = payload.get("history")
        if not history:
            history = [embedding]
        return cls(
            profile_id=str(payload["profile_id"]),
            embedding=embedding,
            num_samples=int(payload.get("num_samples", 1)),
            beers_consumed=int(payload.get("beers_consumed", 0)),
            metadata=dict(payload.get("metadata", {})),
            history=[list(v) for v in history],
            ema_embedding=payload.get("ema_embedding"),
        )


class ProfileStore:
    """Lightweight JSON-backed storage for per-person embeddings."""

    def __init__(
        self,
        path: Path | str = "profiles/profiles.json",
        window_size: int = 5,
        ema_alpha: Optional[float] = 0.2,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if window_size < 0:
            raise ValueError("window_size must be >= 0")
        self.window_size = window_size
        if ema_alpha is not None:
            if not (0 <= ema_alpha <= 1):
                raise ValueError("ema_alpha must be within [0, 1]")
            if ema_alpha == 0:
                ema_alpha = None
        self.ema_alpha = ema_alpha
        self._profiles: Dict[str, Profile] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._profiles = {}
            return
        if not self.path.read_text().strip():
            self._profiles = {}
            return
        data = json.loads(self.path.read_text())
        self._profiles = {
            entry["profile_id"]: Profile.from_dict(entry) for entry in data
        }

    def save(self) -> None:
        payload = [profile.to_dict() for profile in self._profiles.values()]
        self.path.write_text(json.dumps(payload, indent=2))

    @property
    def profiles(self) -> List[Profile]:
        return list(self._profiles.values())

    def get_profile(self, profile_id: str) -> Optional[Profile]:
        return self._profiles.get(profile_id)

    def delete_profile(self, profile_id: str) -> bool:
        """Remove a profile by ID if it exists."""
        if profile_id not in self._profiles:
            return False
        del self._profiles[profile_id]
        self.save()
        return True

    def increment_beers(self, profile_id: str, delta: int = 1) -> int:
        """Increment the beers counter for a profile and persist."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise KeyError(f"Unknown profile_id {profile_id}")
        profile.beers_consumed = max(0, int(profile.beers_consumed) + int(delta))
        self.save()
        return profile.beers_consumed

    def _cosine_similarity(
        self, embedding_a: np.ndarray, embedding_b: np.ndarray
    ) -> float:
        denom = np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
        if denom == 0:
            return 0.0
        return float(np.dot(embedding_a, embedding_b) / denom)

    def _reference_vectors(self, profile: Profile) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        if profile.embedding:
            vectors.append(np.asarray(profile.embedding, dtype=np.float32))
        if profile.ema_embedding is not None:
            vectors.append(np.asarray(profile.ema_embedding, dtype=np.float32))
        return vectors or [
            np.asarray(profile.embedding, dtype=np.float32)
        ]  # Fallback for malformed data

    def _update_ema(self, profile: Profile, embedding: np.ndarray) -> None:
        if self.ema_alpha is None:
            return
        if profile.ema_embedding is None:
            profile.ema_embedding = embedding.tolist()
            return
        prev = np.asarray(profile.ema_embedding, dtype=np.float32)
        updated = (1 - self.ema_alpha) * prev + self.ema_alpha * embedding
        profile.ema_embedding = updated.tolist()

    def similarity_to_profile(self, profile: Profile, embedding: np.ndarray) -> float:
        vectors = self._reference_vectors(profile)
        best = -1.0
        for vector in vectors:
            score = self._cosine_similarity(vector, embedding)
            if score > best:
                best = score
        return best

    def find_match(
        self, embedding: np.ndarray, threshold: float
    ) -> Optional[Tuple[str, float]]:
        best_score = -1.0
        best_id: Optional[str] = None
        for profile in self._profiles.values():
            score = self.similarity_to_profile(profile, embedding)
            if score > threshold and score > best_score:
                best_id = profile.profile_id
                best_score = score
        if best_id is None:
            return None
        return best_id, best_score

    def register_embedding(
        self,
        embedding: np.ndarray,
        profile_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        if profile_id is None:
            profile_id = f"profile_{len(self._profiles) + 1:04d}"
        profile = self._profiles.get(profile_id)
        if profile is None:
            profile = Profile(
                profile_id=profile_id,
                embedding=embedding.tolist(),
                num_samples=1,
                metadata=metadata or {},
                history=[embedding.tolist()],
                ema_embedding=embedding.tolist() if self.ema_alpha is not None else None,
            )
            self._profiles[profile_id] = profile
        else:
            reference = profile.history or []
            reference.append(embedding.tolist())
            if self.window_size > 0 and len(reference) > self.window_size:
                reference = reference[-self.window_size :]
            profile.history = reference
            stacked = np.asarray(reference, dtype=np.float32)
            profile.embedding = stacked.mean(axis=0).tolist()
            profile.num_samples += 1
            if metadata:
                profile.metadata.update(metadata)
            self._update_ema(profile, embedding)
        if profile.ema_embedding is None and self.ema_alpha is not None:
            profile.ema_embedding = embedding.tolist()
        self.save()
        return profile_id
