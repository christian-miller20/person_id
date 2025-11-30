from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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


class ProfileTracker:
    """Runs YOLO to detect people and stores embeddings for future matches."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        profiles_path: str | Path = "profiles/profiles.json",
        device: str = "cpu",
        encoder: str = "mobilenet_v3_small",
        conf: float = 0.5,
        match_threshold: float = 0.55,
        profile_window: int = 5,
        enable_cup_detection: bool = True,
        force_profile_id: Optional[str] = None,
        warmup_threshold: Optional[float] = None,
        warmup_frames: int = 0,
    ) -> None:
        self.model = YOLO(model_name)
        self.extractor = FeatureExtractor(device=device, encoder=encoder)
        self.store = ProfileStore(profiles_path, window_size=profile_window)
        self.conf = conf
        self.match_threshold = match_threshold
        self.drink_labels = {"cup", "wine glass", "bottle"}
        self.cup_iou_threshold = 0.15
        self.enable_cup_detection = enable_cup_detection
        self.force_profile_id = force_profile_id
        self.warmup_threshold = warmup_threshold
        self.warmup_frames = max(0, warmup_frames)

    def _process_boxes(
        self, frame: np.ndarray, results, threshold: float
    ) -> Tuple[List[TrackEvent], List[Tuple[int, int, int, int]]]:
        boxes = results.boxes
        events: List[TrackEvent] = []
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
            if self.force_profile_id:
                profile_id = self.force_profile_id
                profile = self.store.get_profile(profile_id)
                if profile is not None:
                    reference = np.asarray(profile.embedding, dtype=np.float32)
                    match_score = self.store._cosine_similarity(reference, embedding)
                else:
                    match_score = 1.0
                self.store.register_embedding(embedding, profile_id=profile_id)
            else:
                match = self.store.find_match(embedding, threshold=threshold)
                if match is None:
                    profile_id = self.store.register_embedding(embedding)
                    match_score = 1.0
                else:
                    profile_id, match_score = match
                    self.store.register_embedding(embedding, profile_id=profile_id)
            events.append(
                TrackEvent(
                    frame_idx=-1,  # Patched in by caller
                    profile_id=profile_id,
                    confidence=min(match_score, 1.0),
                    box=(x1, y1, x2, y2),
                )
            )
        return events, cup_boxes

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
        cup_flag = " cup" if event.has_cup else ""
        label = f"{event.profile_id} {event.confidence:.2f}{cup_flag}"
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
            events, cup_boxes = self._process_boxes(frame, model_results, threshold)
            self._assign_cups(events, cup_boxes)
            for event in events:
                event.frame_idx = frame_counter
                print(
                    f"[bold green]frame {event.frame_idx:06d}[/] -> {event.profile_id} "
                    f"(similarity={event.confidence:.2f})"
                    + (" [cup]" if event.has_cup else "")
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
