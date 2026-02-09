import numpy as np

from person_id_pi.beverage_pipeline import BeveragePipeline
from person_id_pi.beverage_types import BeverageDetection
from person_id_pi.face_pipeline import FrameTrackAnnotation


class _FakeVideoCapture:
    def __init__(self, frames):
        self._frames = frames
        self._idx = 0

    def isOpened(self):
        return True

    def read(self):
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame

    def get(self, prop):
        # Minimal properties used by build_events.
        if int(prop) == 3:  # CAP_PROP_FRAME_WIDTH
            return float(self._frames[0].shape[1]) if self._frames else 0.0
        if int(prop) == 4:  # CAP_PROP_FRAME_HEIGHT
            return float(self._frames[0].shape[0]) if self._frames else 0.0
        return 0.0

    def release(self):
        return None


class _FakeDetector:
    def __init__(self, by_frame):
        self._by_frame = by_frame
        self._idx = 0

    def detect(self, frame):
        detections = self._by_frame[self._idx] if self._idx < len(self._by_frame) else []
        self._idx += 1
        return detections


def test_build_events_emits_one_event_for_persistent_object_track(monkeypatch):
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    monkeypatch.setattr(
        "person_id_pi.beverage_pipeline.cv2.VideoCapture",
        lambda source: _FakeVideoCapture(frames),
    )
    detector = _FakeDetector(
        [
            [BeverageDetection(bbox=(10, 10, 30, 30), label="can", score=0.8)],
            [BeverageDetection(bbox=(11, 11, 31, 31), label="can", score=0.85)],
            [BeverageDetection(bbox=(12, 12, 32, 32), label="can", score=0.82)],
        ]
    )
    pipeline = BeveragePipeline(detector=detector, object_iou_threshold=0.1)

    frame_annotations = [
        [FrameTrackAnnotation(track_id=7, bbox=(8, 8, 35, 35))],
        [FrameTrackAnnotation(track_id=7, bbox=(8, 8, 35, 35))],
        [FrameTrackAnnotation(track_id=7, bbox=(8, 8, 35, 35))],
    ]
    events = pipeline.build_events(
        source="dummy.mp4",
        frame_annotations=frame_annotations,
        track_to_user={7: "alice"},
        video_id="vidA",
        count_beers=True,
        count_espressos=False,
    )

    assert len(events) == 1
    assert events[0].user_id == "alice"
    assert events[0].beverage_label == "can"
    assert events[0].track_id == 7


def test_build_events_skips_unresolved_users(monkeypatch):
    frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
    monkeypatch.setattr(
        "person_id_pi.beverage_pipeline.cv2.VideoCapture",
        lambda source: _FakeVideoCapture(frames),
    )
    detector = _FakeDetector(
        [[BeverageDetection(bbox=(10, 10, 30, 30), label="bottle", score=0.9)]]
    )
    pipeline = BeveragePipeline(detector=detector, object_iou_threshold=0.1)

    frame_annotations = [[FrameTrackAnnotation(track_id=5, bbox=(5, 5, 40, 40))]]
    events = pipeline.build_events(
        source="dummy.mp4",
        frame_annotations=frame_annotations,
        track_to_user={},
        video_id="vidB",
        count_beers=True,
        count_espressos=False,
    )

    assert events == []
