from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

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
    ) -> None:
        self.model = YOLO(model_name)
        self.extractor = FeatureExtractor(device=device, encoder=encoder)
        self.store = ProfileStore(profiles_path)
        self.conf = conf
        self.match_threshold = match_threshold

    def _process_boxes(self, frame: np.ndarray, results) -> Iterable[TrackEvent]:
        boxes = results.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:  # YOLO class 0 == person
                continue
            conf = float(box.conf[0])
            if conf < self.conf:
                continue
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1, x2, y2 = self.extractor.preprocess_box(
                (x1, y1, x2, y2), frame.shape
            )
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            embedding = self.extractor(crop)
            match = self.store.find_match(embedding, threshold=self.match_threshold)
            if match is None:
                profile_id = self.store.register_embedding(embedding)
                match_score = 1.0
            else:
                profile_id, match_score = match
                self.store.register_embedding(embedding, profile_id=profile_id)
            yield TrackEvent(
                frame_idx=-1,  # Patched in by caller
                profile_id=profile_id,
                confidence=min(match_score, 1.0),
                box=(x1, y1, x2, y2),
            )

    @staticmethod
    def _draw_event(frame: np.ndarray, event: TrackEvent) -> None:
        """Annotate a frame with the profile ID and similarity score."""
        x1, y1, x2, y2 = event.box
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{event.profile_id} {event.confidence:.2f}"
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
            events = list(self._process_boxes(frame, model_results))
            for event in events:
                event.frame_idx = frame_counter
                print(
                    f"[bold green]frame {event.frame_idx:06d}[/] -> {event.profile_id} "
                    f"(similarity={event.confidence:.2f})"
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
