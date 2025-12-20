from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from rich import print
from rich.progress import track
from ultralytics import YOLO

from .feature_extractor import FeatureExtractor
from .profile_store import ProfileStore


@dataclass
class TrackEvent:
    frame_idx: int
    profile_id: str
    confidence: float
    box: Tuple[int, int, int, int]
    has_cup: bool = False
    beers_consumed: int = 0


@dataclass
class PersonDetection:
    box: Tuple[int, int, int, int]
    embedding: np.ndarray


@dataclass
class ActiveTrack:
    track_id: int
    profile_id: str
    last_box: Tuple[int, int, int, int]
    last_seen: int


class ProfileTracker:
    """Runs YOLO to detect people and stores embeddings for future matches."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        profiles_path: str | Path = "profiles/profiles.json",
        device: str = "cpu",
        encoder: str = "mobilenet_v3_small",
        reid_model_path: Optional[str | Path] = None,
        conf: float = 0.5,
        match_threshold: float = 0.55,
        profile_window: int = 5,
        enable_cup_detection: bool = True,
        enable_temporal_tracking: bool = True,
        track_iou_threshold: float = 0.3,
        track_max_age: int = 10,
        keep_threshold: Optional[float] = None,
        force_profile_id: Optional[str] = None,
        warmup_threshold: Optional[float] = None,
        warmup_frames: int = 0,
        ema_alpha: Optional[float] = 0.2,
    ) -> None:
        self.model = YOLO(model_name)
        self.extractor = FeatureExtractor(
            device=device, encoder=encoder, reid_model_path=reid_model_path
        )
        self.store = ProfileStore(
            profiles_path, window_size=profile_window, ema_alpha=ema_alpha
        )
        self.conf = conf
        self.match_threshold = match_threshold
        self.drink_labels = {"cup", "wine glass", "bottle"}
        self.cup_iou_threshold = 0.15
        self.enable_cup_detection = enable_cup_detection
        self.enable_temporal_tracking = enable_temporal_tracking
        if track_iou_threshold < 0 or track_iou_threshold > 1:
            raise ValueError("track_iou_threshold must be within [0, 1]")
        self.track_iou_threshold = track_iou_threshold
        if track_max_age < 0:
            raise ValueError("track_max_age must be >= 0")
        self.track_max_age = track_max_age
        if keep_threshold is None:
            keep_threshold = max(0.0, match_threshold - 0.1)
        if keep_threshold < 0 or keep_threshold > 1:
            raise ValueError("keep_threshold must be within [0, 1]")
        self.keep_threshold = keep_threshold
        self.force_profile_id = force_profile_id
        self.warmup_threshold = warmup_threshold
        self.warmup_frames = max(0, warmup_frames)
        self._tracks: Dict[int, ActiveTrack] = {}
        self._next_track_id = 1

    def _extract_detections(
        self, frame: np.ndarray, results
    ) -> Tuple[List[PersonDetection], List[Tuple[int, int, int, int]]]:
        boxes = results.boxes
        detections: List[PersonDetection] = []
        cup_boxes: List[Tuple[int, int, int, int]] = []
        names = getattr(results, "names", None) or getattr(
            getattr(self.model, "model", None), "names", {}
        )
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = (
                names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            )
            conf = float(box.conf[0])
            if conf < self.conf:
                continue
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1, x2, y2 = self.extractor.preprocess_box(
                (x1, y1, x2, y2), frame.shape
            )
            if self.enable_cup_detection and cls_name in self.drink_labels:
                cup_boxes.append((x1, y1, x2, y2))
                continue
            if cls_name != "person":
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            embedding = self.extractor(crop)
            detections.append(PersonDetection(box=(x1, y1, x2, y2), embedding=embedding))
        return detections, cup_boxes

    def _prune_tracks(self, frame_idx: int) -> None:
        if self.track_max_age == 0:
            self._tracks = {}
            return
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_idx - track.last_seen > self.track_max_age
        ]
        for track_id in stale:
            del self._tracks[track_id]

    def _greedy_track_assignments(
        self, detections: List[PersonDetection]
    ) -> Dict[int, int]:
        """Return mapping from detection index -> track_id based on best IoU."""
        if not self._tracks or not detections:
            return {}
        candidates: List[Tuple[float, int, int]] = []
        for det_idx, det in enumerate(detections):
            for track_id, track in self._tracks.items():
                iou = self._iou(det.box, track.last_box)
                if iou >= self.track_iou_threshold:
                    candidates.append((iou, det_idx, track_id))
        candidates.sort(reverse=True, key=lambda item: item[0])
        assignments: Dict[int, int] = {}
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, det_idx, track_id in candidates:
            if det_idx in used_dets or track_id in used_tracks:
                continue
            assignments[det_idx] = track_id
            used_dets.add(det_idx)
            used_tracks.add(track_id)
        return assignments

    def _similarity_to_profile_id(self, profile_id: str, embedding: np.ndarray) -> float:
        profile = self.store.get_profile(profile_id)
        if profile is None:
            return -1.0
        return self.store.similarity_to_profile(profile, embedding)

    def _resolve_profile(
        self,
        embedding: np.ndarray,
        threshold: float,
        preferred_profile_id: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Return (profile_id, similarity_score) and update the store."""
        if self.force_profile_id:
            profile_id = self.force_profile_id
            score = self._similarity_to_profile_id(profile_id, embedding)
            if score < 0:
                score = 1.0
            self.store.register_embedding(embedding, profile_id=profile_id)
            return profile_id, min(score, 1.0)

        if preferred_profile_id:
            score = self._similarity_to_profile_id(preferred_profile_id, embedding)
            if score >= self.keep_threshold:
                self.store.register_embedding(embedding, profile_id=preferred_profile_id)
                return preferred_profile_id, min(score, 1.0)

        match = self.store.find_match(embedding, threshold=threshold)
        if match is None:
            profile_id = self.store.register_embedding(embedding)
            return profile_id, 1.0
        profile_id, score = match
        self.store.register_embedding(embedding, profile_id=profile_id)
        return profile_id, min(score, 1.0)

    def _detections_to_events(
        self,
        detections: List[PersonDetection],
        frame_idx: int,
        threshold: float,
    ) -> List[TrackEvent]:
        self._prune_tracks(frame_idx)
        assignments: Dict[int, int] = {}
        if self.enable_temporal_tracking and not self.force_profile_id:
            assignments = self._greedy_track_assignments(detections)

        events: List[TrackEvent] = []
        for det_idx, det in enumerate(detections):
            preferred_profile_id = None
            assigned_track_id = assignments.get(det_idx)
            if assigned_track_id is not None:
                preferred_profile_id = self._tracks[assigned_track_id].profile_id
            profile_id, match_score = self._resolve_profile(
                det.embedding, threshold=threshold, preferred_profile_id=preferred_profile_id
            )
            if assigned_track_id is None:
                assigned_track_id = self._next_track_id
                self._next_track_id += 1
            self._tracks[assigned_track_id] = ActiveTrack(
                track_id=assigned_track_id,
                profile_id=profile_id,
                last_box=det.box,
                last_seen=frame_idx,
            )
            events.append(
                TrackEvent(
                    frame_idx=frame_idx,
                    profile_id=profile_id,
                    confidence=match_score,
                    box=det.box,
                )
            )
        return events

    def _assign_cups(
        self,
        events: Iterable[TrackEvent],
        cup_boxes: Iterable[Tuple[int, int, int, int]],
    ) -> None:
        if not self.enable_cup_detection:
            return
        for event in events:
            for cup_box in cup_boxes:
                if self._iou(event.box, cup_box) >= self.cup_iou_threshold:
                    event.has_cup = True
                    break

    @staticmethod
    def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
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
    def _draw_event(frame: np.ndarray, event: TrackEvent) -> None:
        """Annotate a frame with the profile ID and similarity score."""
        x1, y1, x2, y2 = event.box
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cup_flag = " beer" if event.has_cup else ""
        label = (
            f"{event.profile_id} beers consumed: {event.beers_consumed}"
            f" sim {event.confidence:.2f}{cup_flag}"
        )
        text_origin = (x1, max(15, y1 - 10))
        cv2.putText(
            frame,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    def process_video(
        self,
        source: str,
        save_video_path: Optional[str] = None,
        limit_frames: Optional[int] = None,
        count_beers: bool = False,
        min_primary_frames: int = 3,
        min_beer_frames: int = 1,
    ) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source {source}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        output_writer = None
        if save_video_path:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            Path(save_video_path).parent.mkdir(parents=True, exist_ok=True)
            output_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))

        frame_counter = 0
        primary_counts: Dict[str, int] = {}
        beer_counts: Dict[str, int] = {}
        incremented_profile_id: Optional[str] = None
        frame_iter = range(total_frames) if total_frames > 0 else iter(int, 1)
        for _ in track(frame_iter, description="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            model_results = self.model(frame, verbose=False)[0]
            threshold = self.match_threshold
            if (
                self.warmup_threshold is not None
                and frame_counter < self.warmup_frames
            ):
                threshold = self.warmup_threshold
            detections, cup_boxes = self._extract_detections(frame, model_results)
            events = self._detections_to_events(
                detections, frame_idx=frame_counter, threshold=threshold
            )
            self._assign_cups(events, cup_boxes)

            if count_beers and events:
                for event in events:
                    primary_counts[event.profile_id] = primary_counts.get(event.profile_id, 0) + 1
                    if event.has_cup:
                        beer_counts[event.profile_id] = beer_counts.get(event.profile_id, 0) + 1
                candidate_profile_id = max(primary_counts, key=primary_counts.get)
                if (
                    incremented_profile_id is None
                    and primary_counts.get(candidate_profile_id, 0) >= max(1, min_primary_frames)
                    and beer_counts.get(candidate_profile_id, 0) >= max(1, min_beer_frames)
                ):
                    new_total = self.store.increment_beers(candidate_profile_id, 1)
                    incremented_profile_id = candidate_profile_id
                    print(
                        f"[bold yellow]beer +1[/] -> {candidate_profile_id} "
                        f"(beers_consumed={new_total})"
                    )

            for event in events:
                profile = self.store.get_profile(event.profile_id)
                event.beers_consumed = profile.beers_consumed if profile else 0
                print(
                    f"[bold green]frame {event.frame_idx:06d}[/] -> {event.profile_id} "
                    f"(similarity={event.confidence:.2f})"
                    + (" [beer]" if event.has_cup else "")
                )
            for event in events:
                self._draw_event(frame, event)
            if output_writer is not None:
                output_writer.write(frame)
            frame_counter += 1
            if limit_frames and frame_counter >= limit_frames:
                break

        cap.release()
        if output_writer is not None:
            output_writer.release()
        print(f"[bold blue]Stored {len(self.store.profiles)} profiles[/]")
