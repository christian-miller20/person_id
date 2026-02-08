from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .face_embedder import FaceEmbedder
from .face_types import FaceEmbedding, IdentityDecision, TrackletEmbedding
from .identity_engine import IdentityEngine


@dataclass
class ActiveTrack:
    track_id: int
    last_bbox: Tuple[int, int, int, int]
    last_seen: int
    embeddings: List[FaceEmbedding] = field(default_factory=list)


@dataclass
class FrameTrackAnnotation:
    track_id: int
    bbox: Tuple[int, int, int, int]


class FacePipeline:
    def __init__(
        self,
        embedder: FaceEmbedder,
        identity: IdentityEngine,
        track_iou_threshold: float = 0.3,
        track_max_age: int = 10,
    ) -> None:
        self.embedder = embedder
        self.identity = identity
        self.track_iou_threshold = track_iou_threshold
        self.track_max_age = track_max_age

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

    def _prune_tracks(
        self, tracks: Dict[int, ActiveTrack], frame_idx: int
    ) -> Dict[int, ActiveTrack]:
        if self.track_max_age == 0:
            stale_tracks = dict(tracks)
            tracks.clear()
            return stale_tracks
        stale = [
            track_id
            for track_id, track in tracks.items()
            if frame_idx - track.last_seen > self.track_max_age
        ]
        stale_tracks: Dict[int, ActiveTrack] = {}
        for track_id in stale:
            stale_tracks[track_id] = tracks.pop(track_id)
        return stale_tracks

    def _open_video_source(self, source: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source {source}")
        return cap

    def _open_video_writer(
        self,
        output_path: Path | str,
        fps: float,
        width: int,
        height: int,
    ) -> cv2.VideoWriter:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # VS Code's media preview is stricter than QuickTime; prefer H.264 tags first.
        suffix = output_path.suffix.lower()
        if suffix == ".mp4":
            codec_candidates = ("avc1", "H264", "mp4v")
        elif suffix == ".avi":
            codec_candidates = ("MJPG", "XVID")
        else:
            codec_candidates = ("mp4v",)
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

    def _draw_box_with_label(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str,
        accepted: bool,
    ) -> None:
        x1, y1, x2, y2 = bbox
        color = (0, 200, 0) if accepted else (0, 140, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    def _build_tracklets_with_annotations(
        self,
        source: str,
        limit_frames: Optional[int] = None,
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Dict[int, TrackletEmbedding], List[List[FrameTrackAnnotation]]]:
        cap = self._open_video_source(source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if limit_frames and total_frames > 0:
            total_frames = min(total_frames, limit_frames)
        tracks: Dict[int, ActiveTrack] = {}
        finalized_tracklets: Dict[int, TrackletEmbedding] = {}
        frame_annotations: List[List[FrameTrackAnnotation]] = []
        next_track_id = 1
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces = self.embedder.detect(frame)
            stale_tracks = self._prune_tracks(tracks, frame_idx)
            for stale_track_id, stale_track in stale_tracks.items():
                finalized_tracklets[stale_track_id] = self.identity.aggregate_tracklet(
                    stale_track.embeddings
                )
            assignments: Dict[int, int] = {}
            used_tracks: set[int] = set()
            for face_idx, face in enumerate(faces):
                best_iou = 0.0
                best_track_id = None
                for track_id, track in tracks.items():
                    if track_id in used_tracks:
                        continue
                    iou = self._iou(face.bbox, track.last_bbox)
                    if iou >= self.track_iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id
                if best_track_id is not None:
                    assignments[face_idx] = best_track_id
                    used_tracks.add(best_track_id)

            this_frame_annotations: List[FrameTrackAnnotation] = []
            if total_frames:
                self._emit(
                    verbose,
                    log_fn,
                    f"frame {frame_idx + 1}/{total_frames} -> faces={len(faces)}",
                )
            else:
                self._emit(
                    verbose,
                    log_fn,
                    f"frame {frame_idx:06d} -> faces={len(faces)}",
                )
            for face_idx, face in enumerate(faces):
                track_id = assignments.get(face_idx)
                if track_id is None:
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = ActiveTrack(
                        track_id=track_id,
                        last_bbox=face.bbox,
                        last_seen=frame_idx,
                    )
                track = tracks[track_id]
                track.last_bbox = face.bbox
                track.last_seen = frame_idx
                track.embeddings.append(self.embedder.embed(face))
                this_frame_annotations.append(
                    FrameTrackAnnotation(track_id=track_id, bbox=face.bbox)
                )
                self._emit(
                    verbose,
                    log_fn,
                    "  track={track} quality={quality:.3f} det={det:.3f} "
                    "size={size:.3f} blur={blur:.3f}".format(
                        track=track_id,
                        quality=face.quality,
                        det=face.det_score,
                        size=face.size_score,
                        blur=face.blur_score,
                    ),
                )

            frame_annotations.append(this_frame_annotations)
            frame_idx += 1
            if limit_frames and frame_idx >= limit_frames:
                break
        cap.release()
        # Finalize currently-active tracks.
        tracklets = {
            track_id: self.identity.aggregate_tracklet(track.embeddings)
            for track_id, track in tracks.items()
        }
        # Include stale finalized tracks as well.
        tracklets.update(finalized_tracklets)
        return tracklets, frame_annotations

    def write_multi_face_annotations(
        self,
        source: str,
        output_path: Path | str,
        frame_annotations: List[List[FrameTrackAnnotation]],
        decisions: Dict[int, IdentityDecision],
    ) -> None:
        cap = self._open_video_source(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = self._open_video_writer(output_path, fps, width, height)

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotations = (
                frame_annotations[frame_idx]
                if frame_idx < len(frame_annotations)
                else []
            )
            for item in annotations:
                decision = decisions.get(item.track_id)
                accepted = bool(decision and decision.accepted and decision.user_id)
                label = decision.user_id if accepted and decision else "unknown"
                self._draw_box_with_label(
                    frame, item.bbox, str(label), accepted=accepted
                )
            writer.write(frame)
            frame_idx += 1
            if frame_idx >= len(frame_annotations):
                break
        writer.release()
        cap.release()

    def extract_tracklets_with_annotations_from_video(
        self,
        source: str,
        limit_frames: Optional[int] = None,
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Dict[int, TrackletEmbedding], List[List[FrameTrackAnnotation]]]:
        return self._build_tracklets_with_annotations(
            source=source, limit_frames=limit_frames, verbose=verbose, log_fn=log_fn
        )

    def extract_primary_tracklet_from_video(
        self,
        source: str,
        limit_frames: Optional[int] = None,
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> TrackletEmbedding:
        tracklets_by_id, _ = self.extract_tracklets_with_annotations_from_video(
            source=source, limit_frames=limit_frames, verbose=verbose, log_fn=log_fn
        )
        if not tracklets_by_id:
            return TrackletEmbedding(
                embedding=np.zeros((512,), dtype=np.float32),
                n_used=0,
                dispersion=0.0,
                outliers=0,
            )
        ranked = sorted(
            tracklets_by_id.values(),
            key=lambda t: (t.n_used, -t.dispersion),
            reverse=True,
        )
        return ranked[0]
