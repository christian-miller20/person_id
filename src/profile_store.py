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
    metadata: Dict[str, str] = field(default_factory=dict)
    history: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "embedding": self.embedding,
            "num_samples": self.num_samples,
            "metadata": self.metadata,
            "history": self.history,
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
            metadata=dict(payload.get("metadata", {})),
            history=[list(v) for v in history],
        )


class ProfileStore:
    """Lightweight JSON-backed storage for per-person embeddings."""

    def __init__(
        self, path: Path | str = "profiles/profiles.json", window_size: int = 5
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if window_size < 0:
            raise ValueError("window_size must be >= 0")
        self.window_size = window_size
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

    def _cosine_similarity(
        self, embedding_a: np.ndarray, embedding_b: np.ndarray
    ) -> float:
        denom = np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
        if denom == 0:
            return 0.0
        return float(np.dot(embedding_a, embedding_b) / denom)

    def find_match(
        self, embedding: np.ndarray, threshold: float
    ) -> Optional[Tuple[str, float]]:
        best_score = -1.0
        best_id: Optional[str] = None
        for profile in self._profiles.values():
            reference = np.asarray(profile.embedding, dtype=np.float32)
            score = self._cosine_similarity(reference, embedding)
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
        self.save()
        return profile_id
