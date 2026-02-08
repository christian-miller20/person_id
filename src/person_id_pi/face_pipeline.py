from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .face_embedder import FaceEmbedder
from .face_types import FaceEmbedding, TrackletEmbedding
from .identity_engine import IdentityEngine


@dataclass
class ClipResult:
    decision_user_id: Optional[str]
    decision_score: float
    decision_margin: float
    accepted: bool
    reason: str
    n_used: int
    dispersion: float


@dataclass
class TrackDecision:
    track_id: int
    decision_user_id: Optional[str]
    decision_score: float
    decision_margin: float
    accepted: bool
    reason: str
    n_used: int
    dispersion: float


@dataclass
class ActiveTrack:
    track_id: int
    last_bbox: Tuple[int, int, int, int]
    last_seen: int
    embeddings: List[FaceEmbedding] = field(default_factory=list)


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

    def _select_face(self, faces: List) -> Optional:
        if not faces:
            return None
        # Single-person mode: choose the best detection by confidence.
        faces.sort(key=lambda f: f.quality, reverse=True)
        return faces[0]

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

    def _prune_tracks(self, tracks: Dict[int, ActiveTrack], frame_idx: int) -> None:
        if self.track_max_age == 0:
            tracks.clear()
            return
        stale = [
            track_id
            for track_id, track in tracks.items()
            if frame_idx - track.last_seen > self.track_max_age
        ]
        for track_id in stale:
            del tracks[track_id]

    def _collect_embeddings_from_frames(
        self, frames: Iterable[np.ndarray], verbose: bool = False
    ) -> List[FaceEmbedding]:
        embeddings: List[FaceEmbedding] = []
        for idx, frame in enumerate(frames):
            faces = self.embedder.detect(frame)
            face = self._select_face(faces)
            if face is None:
                if verbose:
                    print(f"frame {idx:06d} -> faces=0")
                continue
            if verbose:
                print(
                    "frame {frame} -> faces={faces} quality={quality:.3f} det={det:.3f} "
                    "size={size:.3f} blur={blur:.3f}".format(
                        frame=f"{idx:06d}",
                        faces=len(faces),
                        quality=face.quality,
                        det=face.det_score,
                        size=face.size_score,
                        blur=face.blur_score,
                    )
                )
            embeddings.append(self.embedder.embed(face))
        return embeddings

    def process_frames(self, frames: Iterable[np.ndarray], verbose: bool = False) -> ClipResult:
        embeddings = self._collect_embeddings_from_frames(frames, verbose=verbose)
        tracklet = self.identity.aggregate_tracklet(embeddings)
        decision = self.identity.match(tracklet)
        return ClipResult(
            decision_user_id=decision.user_id,
            decision_score=decision.score,
            decision_margin=decision.margin,
            accepted=decision.accepted,
            reason=decision.reason,
            n_used=tracklet.n_used,
            dispersion=tracklet.dispersion,
        )

    def extract_tracklets_from_video(
        self,
        source: str,
        limit_frames: Optional[int] = None,
        verbose: bool = False,
    ) -> List[TrackletEmbedding]:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source {source}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if limit_frames and total_frames > 0:
            total_frames = min(total_frames, limit_frames)
        tracks: Dict[int, ActiveTrack] = {}
        next_track_id = 1
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces = self.embedder.detect(frame)
            self._prune_tracks(tracks, frame_idx)
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
            if verbose:
                if total_frames:
                    print(f"frame {frame_idx + 1}/{total_frames} -> faces={len(faces)}")
                else:
                    print(f"frame {frame_idx:06d} -> faces={len(faces)}")
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
                if verbose:
                    print(
                        "  track={track} quality={quality:.3f} det={det:.3f} "
                        "size={size:.3f} blur={blur:.3f}".format(
                            track=track_id,
                            quality=face.quality,
                            det=face.det_score,
                            size=face.size_score,
                            blur=face.blur_score,
                        )
                    )
            frame_idx += 1
            if limit_frames and frame_idx >= limit_frames:
                break
        cap.release()
        tracklets: List[TrackletEmbedding] = []
        for track in tracks.values():
            tracklets.append(self.identity.aggregate_tracklet(track.embeddings))
        return tracklets

    def extract_tracklet_from_video(
        self,
        source: str,
        limit_frames: Optional[int] = None,
        verbose: bool = False,
    ) -> TrackletEmbedding:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source {source}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if limit_frames and total_frames > 0:
            total_frames = min(total_frames, limit_frames)
        embeddings: List[FaceEmbedding] = []
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces = self.embedder.detect(frame)
            face = self._select_face(faces)
            if face is None:
                if verbose:
                    if total_frames:
                        print(f"frame {count + 1}/{total_frames} -> faces=0")
                    else:
                        print(f"frame {count:06d} -> faces=0")
            else:
                if verbose:
                    if total_frames:
                        print(
                            "frame {frame}/{total} -> faces={faces} quality={quality:.3f} "
                            "det={det:.3f} size={size:.3f} blur={blur:.3f}".format(
                                frame=count + 1,
                                total=total_frames,
                                faces=len(faces),
                                quality=face.quality,
                                det=face.det_score,
                                size=face.size_score,
                                blur=face.blur_score,
                            )
                        )
                    else:
                        print(
                            "frame {frame} -> faces={faces} quality={quality:.3f} "
                            "det={det:.3f} size={size:.3f} blur={blur:.3f}".format(
                                frame=f"{count:06d}",
                                faces=len(faces),
                                quality=face.quality,
                                det=face.det_score,
                                size=face.size_score,
                                blur=face.blur_score,
                            )
                        )
                embeddings.append(self.embedder.embed(face))
            count += 1
            if limit_frames and count >= limit_frames:
                break
        cap.release()
        return self.identity.aggregate_tracklet(embeddings)

    def process_video(
        self,
        source: str,
        limit_frames: Optional[int] = None,
        verbose: bool = False,
    ) -> ClipResult:
        tracklet = self.extract_tracklet_from_video(
            source=source, limit_frames=limit_frames, verbose=verbose
        )
        decision = self.identity.match(tracklet)
        return ClipResult(
            decision_user_id=decision.user_id,
            decision_score=decision.score,
            decision_margin=decision.margin,
            accepted=decision.accepted,
            reason=decision.reason,
            n_used=tracklet.n_used,
            dispersion=tracklet.dispersion,
        )
