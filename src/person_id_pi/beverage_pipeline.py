from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .beverage_detector import BeverageDetector
from .beverage_types import BeverageDetection, BeverageEvent, BeverageLabel
from .face_pipeline import FrameTrackAnnotation


@dataclass
class ActiveObjectTrack:
    """State for one beverage object track across frames."""

    object_track_id: int
    label: BeverageLabel
    last_bbox: Tuple[int, int, int, int]
    last_seen: int
    best_score: float
    first_frame_idx: int
    seen_count: int
    assigned_person_track_id: Optional[int] = None


@dataclass(frozen=True)
class FrameBeverageAssociation:
    """Association candidate between one beverage detection and one person track."""

    frame_idx: int
    object_track_id: int
    person_track_id: int
    beverage_label: BeverageLabel
    score: float
    bbox: Tuple[int, int, int, int]


class BeveragePipeline:
    """Detect, track, associate, and emit distinct beverage events per user."""

    def __init__(
        self,
        detector: BeverageDetector,
        object_iou_threshold: float = 0.3,
        object_max_age: int = 10,
        association_max_dist_norm: float = 0.25,
        object_min_seen_frames: int = 5,
        event_cooldown_sec: float = 10.0,
    ) -> None:
        self.detector = detector
        self.object_iou_threshold = object_iou_threshold
        self.object_max_age = object_max_age
        self.association_max_dist_norm = association_max_dist_norm
        self.object_min_seen_frames = max(1, int(object_min_seen_frames))
        self.event_cooldown_sec = max(0.0, float(event_cooldown_sec))

    @staticmethod
    def _emit(
        verbose: bool, log_fn: Optional[Callable[[str], None]], message: str
    ) -> None:
        """Emit verbose logs either via callback or stdout."""
        if not verbose:
            return
        if log_fn is not None:
            log_fn(message)
            return
        print(message)

    @staticmethod
    def _open_video_writer(
        output_path: Path | str, fps: float, width: int, height: int
    ) -> cv2.VideoWriter:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        codec_candidates = ("avc1", "mp4v")
        for codec in codec_candidates:
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (width, height),
            )
            if writer.isOpened():
                return writer
            writer.release()
        raise RuntimeError(
            f"Unable to open output writer for {output_path} with codecs {codec_candidates}"
        )

    @staticmethod
    def _draw_beverage_box(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        display_label: str,
    ) -> None:
        x1, y1, x2, y2 = bbox
        color = (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            display_label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _display_label(label: BeverageLabel) -> str:
        if label in {"cup", "can", "bottle"}:
            return "BEER"
        if label == "espresso_shot":
            return "ESPRESSO"
        return str(label).upper()

    @staticmethod
    def _iou(
        box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]
    ) -> float:
        """Intersection-over-Union for axis-aligned bounding boxes."""
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
        """Center point of bbox in image coordinates."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _dist_norm(
        center_a: Tuple[float, float],
        center_b: Tuple[float, float],
        frame_w: int,
        frame_h: int,
    ) -> float:
        """Euclidean center distance normalized by frame diagonal."""
        denom = float(max(1.0, np.hypot(frame_w, frame_h)))
        return float(
            np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]) / denom
        )

    def _prune_object_tracks(
        self, tracks: Dict[int, ActiveObjectTrack], frame_idx: int
    ) -> Dict[int, ActiveObjectTrack]:
        """Drop stale object tracks that exceeded max age."""
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
        """Gate labels by CLI counting mode."""
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
        """Associate beverage to nearest person track if within distance threshold."""
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
        """Stable deterministic ID used for idempotent event persistence."""
        base = f"{video_id}|{object_track_id}|{first_frame_idx}|{user_id}|{label}"
        return sha1(base.encode("utf-8")).hexdigest()[:16]

    def _match_detections_to_object_tracks(
        self,
        detections: List[BeverageDetection],
        tracks: Dict[int, ActiveObjectTrack],
        next_object_track_id: int,
        frame_idx: int,
    ) -> int:
        """Greedy IoU matching from detections to active object tracks."""
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
                # Start a new object track when no suitable IoU match is found.
                tracks[next_object_track_id] = ActiveObjectTrack(
                    object_track_id=next_object_track_id,
                    label=detection.label,
                    last_bbox=detection.bbox,
                    last_seen=frame_idx,
                    best_score=detection.score,
                    first_frame_idx=frame_idx,
                    seen_count=1,
                )
                used_track_ids.add(next_object_track_id)
                next_object_track_id += 1
                continue
            track = tracks[best_track_id]
            # Continue an existing object track and accumulate evidence quality.
            track.last_bbox = detection.bbox
            track.last_seen = frame_idx
            track.best_score = max(track.best_score, detection.score)
            track.seen_count += 1
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
        # Convert cooldown seconds into frame units so dedupe is frame-local.
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._emit(verbose, log_fn, f"beverage_video_fps fps={fps:.3f}")
        if fps <= 0.0:
            fps = 30.0
        cooldown_frames = int(round(self.event_cooldown_sec * fps))
        resolved_video_id = video_id or Path(source).name

        events: List[BeverageEvent] = []
        object_tracks: Dict[int, ActiveObjectTrack] = {}
        emitted_object_track_ids: set[int] = set()
        # Per-user/per-label cooldown memory to suppress near-duplicate events.
        last_emitted_frame_by_user_label: Dict[Tuple[str, BeverageLabel], int] = {}
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
                if object_track.seen_count < self.object_min_seen_frames:
                    self._emit(
                        verbose,
                        log_fn,
                        f"beverage_reject frame={frame_idx} object_track={object_track_id} "
                        f"reason=min_seen seen_count={object_track.seen_count} "
                        f"required={self.object_min_seen_frames}",
                    )
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
                last_emitted = last_emitted_frame_by_user_label.get(
                    (user_id, object_track.label)
                )
                if (
                    last_emitted is not None
                    and cooldown_frames > 0
                    and frame_idx - last_emitted <= cooldown_frames
                ):
                    # This track would double-count a recent identical user+label event.
                    self._emit(
                        verbose,
                        log_fn,
                        f"beverage_reject frame={frame_idx} object_track={object_track_id} "
                        f"reason=cooldown user_id={user_id} label={object_track.label} "
                        f"delta_frames={frame_idx - last_emitted} cooldown_frames={cooldown_frames}",
                    )
                    emitted_object_track_ids.add(object_track_id)
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
                # Remember emission time for cooldown dedupe on future tracks.
                last_emitted_frame_by_user_label[(user_id, object_track.label)] = (
                    frame_idx
                )

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

    def annotate_video(
        self,
        source: str,
        output_path: Path | str,
        count_beers: bool = True,
        count_espressos: bool = False,
        limit_frames: Optional[int] = None,
        hold_frames: int = 45,
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Draw beverage detections onto video with short-lived persistence."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source {source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        writer = self._open_video_writer(
            output_path=output_path, fps=fps, width=width, height=height
        )

        frame_idx = 0
        resolved_hold_frames = max(0, int(hold_frames))
        active_overlays: Dict[
            int, Tuple[BeverageLabel, Tuple[int, int, int, int], int]
        ] = {}
        next_overlay_id = 1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if limit_frames is not None and frame_idx >= limit_frames:
                break

            detections = self.detector.detect(frame)
            matched_overlay_ids: set[int] = set()
            for detection in detections:
                if not self._is_countable_label(
                    detection.label,
                    count_beers=count_beers,
                    count_espressos=count_espressos,
                ):
                    continue
                # Re-link to an existing overlay track when possible so labels persist.
                best_overlay_id: Optional[int] = None
                best_iou = 0.0
                for overlay_id, (label, bbox, last_seen) in active_overlays.items():
                    if overlay_id in matched_overlay_ids:
                        continue
                    if label != detection.label:
                        continue
                    if frame_idx - last_seen > resolved_hold_frames:
                        continue
                    iou = self._iou(detection.bbox, bbox)
                    if iou >= self.object_iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_overlay_id = overlay_id
                if best_overlay_id is None:
                    best_overlay_id = next_overlay_id
                    next_overlay_id += 1
                active_overlays[best_overlay_id] = (
                    detection.label,
                    detection.bbox,
                    frame_idx,
                )
                matched_overlay_ids.add(best_overlay_id)

            stale_overlay_ids = [
                overlay_id
                for overlay_id, (_, _, last_seen) in active_overlays.items()
                if frame_idx - last_seen > resolved_hold_frames
            ]
            for overlay_id in stale_overlay_ids:
                active_overlays.pop(overlay_id, None)

            for label, bbox, _ in active_overlays.values():
                self._draw_beverage_box(
                    frame=frame,
                    bbox=bbox,
                    display_label=self._display_label(label),
                )
            writer.write(frame)
            frame_idx += 1
        writer.release()
        cap.release()

    def annotate_video_in_place(
        self,
        video_path: Path | str,
        count_beers: bool = True,
        count_espressos: bool = False,
        limit_frames: Optional[int] = None,
        hold_frames: int = 45,
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Overlay beverage boxes onto an existing video file."""
        target = Path(video_path)
        with NamedTemporaryFile(
            suffix=target.suffix or ".mp4",
            dir=str(target.parent),
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        self.annotate_video(
            source=str(target),
            output_path=tmp_path,
            count_beers=count_beers,
            count_espressos=count_espressos,
            limit_frames=limit_frames,
            hold_frames=hold_frames,
            verbose=verbose,
            log_fn=log_fn,
        )
        tmp_path.replace(target)
