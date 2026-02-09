import numpy as np

from person_id_pi.beverage_config import BeverageDetectorConfig
from person_id_pi.beverage_detector import MultiBeverageDetector, YoloBeverageDetector
from person_id_pi.beverage_types import BeverageDetection


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)


class _FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class _FakeModel:
    def __init__(self, results):
        self._results = results

    def __call__(self, frame, verbose=False):
        return self._results


def test_yolo_detector_maps_and_filters_classes():
    names = {
        0: "cup",
        1: "bottle",
        2: "person",
        3: "beer can",
    }
    boxes = _FakeBoxes(
        xyxy=[[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30], [30, 30, 40, 40]],
        conf=[0.9, 0.1, 0.95, 0.8],
        cls=[0, 1, 2, 3],
    )
    model = _FakeModel([_FakeResult(names=names, boxes=boxes)])
    detector = YoloBeverageDetector(
        config=BeverageDetectorConfig(conf_min=0.25),
        model=model,
    )

    detections = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

    assert len(detections) == 2
    assert detections[0].label == "cup"
    assert detections[1].label == "can"
    assert detections[0].bbox == (0, 0, 10, 10)
    assert detections[1].bbox == (30, 30, 40, 40)


def test_yolo_detector_respects_allowed_labels():
    names = {
        0: "cup",
        1: "bottle",
    }
    boxes = _FakeBoxes(
        xyxy=[[0, 0, 10, 10], [10, 10, 20, 20]],
        conf=[0.9, 0.9],
        cls=[0, 1],
    )
    model = _FakeModel([_FakeResult(names=names, boxes=boxes)])
    detector = YoloBeverageDetector(
        config=BeverageDetectorConfig(conf_min=0.25, allowed_labels=("cup",)),
        model=model,
    )

    detections = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].label == "cup"


class _StaticDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, frame):
        return list(self._detections)


def test_multi_detector_dedupes_same_label_overlap():
    d1 = _StaticDetector(
        [
            BeverageDetection(bbox=(10, 10, 30, 30), label="can", score=0.70),
            BeverageDetection(bbox=(50, 50, 80, 80), label="espresso_shot", score=0.65),
        ]
    )
    d2 = _StaticDetector(
        [
            BeverageDetection(bbox=(11, 11, 31, 31), label="can", score=0.90),
            BeverageDetection(bbox=(70, 70, 90, 90), label="can", score=0.50),
        ]
    )
    detector = MultiBeverageDetector([d1, d2], dedupe_iou_threshold=0.5)

    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    # Overlapping can boxes are deduped to the highest confidence instance.
    can_boxes = [d for d in detections if d.label == "can"]
    assert any(d.bbox == (11, 11, 31, 31) and d.score == 0.90 for d in can_boxes)
    # Non-overlapping can and espresso shot should remain.
    assert len(detections) == 3
