from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .beverage_detector import BeverageDetector
from .beverage_types import BeverageDetection, BeverageEvent, BeverageLabel
from .face_pipeline import FrameTrackAnnotation


# ActiveObjectTrack represents a currently tracked beverage object across frames, along with its best detection score and assigned person track if any.
@dataclass
class ActiveObjectTrack:
    object_track_id: int
    label: BeverageLabel
    last_bbox: Tuple[int, int, int, int]
    last_seen: int
    best_score: float
    first_frame_idx: int
    assigned_person_track_id: Optional[int] = None

# FrameBeverageAssociation represents a single association of a detected beverage object with a person track on a specific frame
@dataclass(frozen=True)
class FrameBeverageAssociation:
    frame_idx: int
    object_track_id: int
    person_track_id: int
    beverage_label: BeverageLabel
    score: float
    bbox: Tuple[int, int, int, int]

# BeveragePipeline processes video frames to detect beverages, track them across frames, and associate them with identified person tracks to generate beverage events.
class BeveragePipeline:
    def __init__(
        self,
        detector: BeverageDetector,
        object_iou_threshold: float = 0.3,
        object_max_age: int = 10,
        association_max_dist_norm: float = 0.25,
    ) -> None:
        self.detector = detector
        self.object_iou_threshold = object_iou_threshold
        self.object_max_age = object_max_age
        self.association_max_dist_norm = association_max_dist_norm

    @staticmethod
    def _emit(
        verbose: bool, log_fn: Optional[Callable[[str], None]], message: str
    ) -> None:
        if not verbose:
            return
        if log_fn is not None:
            log_fn(message)
            return
        print(message)

    @staticmethod
    def _iou(
        box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    @staticmethod
    def _center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _dist_norm(
        center_a: Tuple[float, float],
        center_b: Tuple[float, float],
        frame_w: int,
        frame_h: int,
    ) -> float:
        denom = float(max(1.0, np.hypot(frame_w, frame_h)))
        return float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]) / denom)

    def _prune_object_tracks(
        self, tracks: Dict[int, ActiveObjectTrack], frame_idx: int
    ) -> Dict[int, ActiveObjectTrack]:
        if self.object_max_age == 0:
            stale = dict(tracks)
            tracks.clear()
            return stale
        stale_ids = [
            object_track_id
            for object_track_id, track in tracks.items()
            if frame_idx - track.last_seen > self.object_max_age
        ]
        stale: Dict[int, ActiveObjectTrack] = {}
        for object_track_id in stale_ids:
            stale[object_track_id] = tracks.pop(object_track_id)
        return stale

    def _is_countable_label(
        self,
        label: BeverageLabel,
        count_beers: bool,
        count_espressos: bool,
    ) -> bool:
        if count_beers and label in {"cup", "can", "bottle"}:
            return True
        if count_espressos and label == "espresso_shot":
            return True
        return False

    def _find_person_for_detection(
        self,
        detection_bbox: Tuple[int, int, int, int],
        frame_people: List[FrameTrackAnnotation],
        frame_w: int,
        frame_h: int,
    ) -> Optional[int]:
        if not frame_people:
            return None
        det_center = self._center(detection_bbox)
        best_track_id: Optional[int] = None
        best_dist = float("inf")
        for person in frame_people:
            person_center = self._center(person.bbox)
            dist = self._dist_norm(det_center, person_center, frame_w, frame_h)
            if dist < best_dist:
                best_dist = dist
                best_track_id = person.track_id
        if best_track_id is None:
            return None
        if best_dist > self.association_max_dist_norm:
            return None
        return best_track_id

    @staticmethod
    def _make_event_id(
        video_id: str,
        object_track_id: int,
        first_frame_idx: int,
        user_id: str,
        label: BeverageLabel,
    ) -> str:
        base = f"{video_id}|{object_track_id}|{first_frame_idx}|{user_id}|{label}"
        return sha1(base.encode("utf-8")).hexdigest()[:16]

    def _match_detections_to_object_tracks(
        self,
        detections: List[BeverageDetection],
        tracks: Dict[int, ActiveObjectTrack],
        next_object_track_id: int,
        frame_idx: int,
    ) -> int:
        used_track_ids: set[int] = set()
        for detection in detections:
            best_iou = 0.0
            best_track_id: Optional[int] = None
            for object_track_id, track in tracks.items():
                if object_track_id in used_track_ids:
                    continue
                if track.label != detection.label:
                    continue
                iou = self._iou(detection.bbox, track.last_bbox)
                if iou >= self.object_iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track_id = object_track_id
            if best_track_id is None:
                tracks[next_object_track_id] = ActiveObjectTrack(
                    object_track_id=next_object_track_id,
                    label=detection.label,
                    last_bbox=detection.bbox,
                    last_seen=frame_idx,
                    best_score=detection.score,
                    first_frame_idx=frame_idx,
                )
                used_track_ids.add(next_object_track_id)
                next_object_track_id += 1
                continue
            track = tracks[best_track_id]
            track.last_bbox = detection.bbox
            track.last_seen = frame_idx
            track.best_score = max(track.best_score, detection.score)
            used_track_ids.add(best_track_id)
        return next_object_track_id

    def build_events(
        self,
        source: str,
        frame_annotations: List[List[FrameTrackAnnotation]],
        track_to_user: Dict[int, str],
        video_id: Optional[str] = None,
        count_beers: bool = True,
        count_espressos: bool = False,
        limit_frames: Optional[int] = None,
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> List[BeverageEvent]:
        """
        Build distinct beverage events for identified users.

        Distinct policy: one event per object track (first valid assignment to an
        accepted person track).
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source {source}")
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        resolved_video_id = video_id or Path(source).name

        events: List[BeverageEvent] = []
        object_tracks: Dict[int, ActiveObjectTrack] = {}
        emitted_object_track_ids: set[int] = set()
        next_object_track_id = 1
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if limit_frames is not None and frame_idx >= limit_frames:
                break

            self._prune_object_tracks(object_tracks, frame_idx)
            detections = self.detector.detect(frame)
            next_object_track_id = self._match_detections_to_object_tracks(
                detections=detections,
                tracks=object_tracks,
                next_object_track_id=next_object_track_id,
                frame_idx=frame_idx,
            )
            frame_people = (
                frame_annotations[frame_idx]
                if frame_idx < len(frame_annotations)
                else []
            )

            self._emit(
                verbose,
                log_fn,
                f"beverage_frame frame={frame_idx} detections={len(detections)}",
            )

            # Distinct policy: one event per object track at first valid user assignment.
            for object_track_id, object_track in object_tracks.items():
                if object_track.last_seen != frame_idx:
                    continue
                if object_track_id in emitted_object_track_ids:
                    continue
                if not self._is_countable_label(
                    object_track.label,
                    count_beers=count_beers,
                    count_espressos=count_espressos,
                ):
                    continue

                person_track_id = self._find_person_for_detection(
                    detection_bbox=object_track.last_bbox,
                    frame_people=frame_people,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )
                if person_track_id is None:
                    continue
                user_id = track_to_user.get(person_track_id)
                if not user_id:
                    continue

                object_track.assigned_person_track_id = person_track_id
                event_id = self._make_event_id(
                    video_id=resolved_video_id,
                    object_track_id=object_track_id,
                    first_frame_idx=object_track.first_frame_idx,
                    user_id=user_id,
                    label=object_track.label,
                )
                event = BeverageEvent(
                    event_id=event_id,
                    video_id=resolved_video_id,
                    frame_idx=frame_idx,
                    track_id=person_track_id,
                    user_id=user_id,
                    beverage_label=object_track.label,
                    object_track_id=object_track_id,
                    confidence=object_track.best_score,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                events.append(event)
                emitted_object_track_ids.add(object_track_id)

                self._emit(
                    verbose,
                    log_fn,
                    f"beverage_event frame={frame_idx} object_track={object_track_id} "
                    f"person_track={person_track_id} user_id={user_id} "
                    f"label={object_track.label} score={object_track.best_score:.3f}",
                )

            frame_idx += 1

        cap.release()
        return events
