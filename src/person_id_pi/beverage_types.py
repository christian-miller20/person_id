from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

BeverageLabel = Literal["cup", "can", "bottle", "espresso_shot"]
BeerLabels = {"can", "bottle", "cup"}  # Assuming "cup" can also represent beer in some contexts
EspressoLabels = {"espresso_shot"}

@dataclass(frozen=True)
class BeverageDetection:
    # (x1, y1, x2, y2) bounding box in image coordinates.
    bbox: Tuple[int, int, int, int]
    label: BeverageLabel
    score: float  # Confidence score from the beverage detector (e.g., YOLOv8)

@dataclass(frozen=True)
class BeverageEvent:
    # Deterministic id for idempotent persistence.
    event_id: str
    # Logical video identifier (usually source path or normalized stem).
    video_id: str
    # Frame index where the event was first observed.
    frame_idx: int
    # Person track id from face pipeline.
    track_id: int
    # Resolved accepted user id.
    user_id: str
    # Beverage class.
    beverage_label: BeverageLabel
    # Object tracker id for distinct-event policy.
    object_track_id: int
    # Confidence for this event (typically detection score at first observation).
    confidence: float
    # UTC timestamp string (ISO-8601).
    timestamp_utc: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "event_id": self.event_id,
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "track_id": self.track_id,
            "user_id": self.user_id,
            "beverage_label": self.beverage_label,
            "object_track_id": self.object_track_id,
            "confidence": self.confidence,
            "timestamp_utc": self.timestamp_utc,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "BeverageEvent":
        return cls(
            event_id=str(payload["event_id"]),
            video_id=str(payload["video_id"]),
            frame_idx=int(payload["frame_idx"]),
            track_id=int(payload["track_id"]),
            user_id=str(payload["user_id"]),
            beverage_label=str(payload["beverage_label"]),  # type: ignore[arg-type]
            object_track_id=int(payload["object_track_id"]),
            confidence=float(payload["confidence"]),
            timestamp_utc=str(payload["timestamp_utc"]),
        )

@dataclass(frozen=True)
class PerUserBeverageSummary:
    user_id: str
    beers_in_video: int
    beers_total: int
    espressos_in_video: int = 0
    espressos_total: int = 0
