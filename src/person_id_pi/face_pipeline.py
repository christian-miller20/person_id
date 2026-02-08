from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np

from .face_embedder import FaceEmbedder
from .face_types import FaceEmbedding
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


class FacePipeline:
    def __init__(self, embedder: FaceEmbedder, identity: IdentityEngine) -> None:
        self.embedder = embedder
        self.identity = identity

    def _select_face(self, faces: List) -> Optional:
        if not faces:
            return None
        # Single-person mode: choose the best detection by confidence.
        faces.sort(key=lambda f: f.quality, reverse=True)
        return faces[0]

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

    def extract_tracklet_from_video(
        self,
        source: str,
        limit_frames: Optional[int] = None,
        verbose: bool = False,
    ):
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
