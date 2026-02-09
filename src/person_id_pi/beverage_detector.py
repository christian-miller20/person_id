from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .beverage_config import BeverageDetectorConfig
from .beverage_types import BeverageDetection, BeverageLabel


LABEL_ALIASES = {
    "cup": "cup",
    "can": "can",
    "bottle": "bottle",
    "beer can": "can",
    "soda can": "can",
    "soft drink can": "can",
    "espresso shot": "espresso_shot",
    "espresso_shot": "espresso_shot",
}


class BeverageDetector(Protocol):
    def detect(self, frame: np.ndarray) -> List[BeverageDetection]:
        ...


def _normalize_label(raw_label: str) -> Optional[BeverageLabel]:
    normalized = LABEL_ALIASES.get(raw_label.lower().strip())
    if normalized is None:
        return None
    return normalized  # type: ignore[return-value]


def _to_list(values: object) -> List[float]:
    if hasattr(values, "tolist"):
        result = values.tolist()
        if isinstance(result, list):
            if result and isinstance(result[0], list):
                return [float(v) for row in result for v in row]
            return [float(v) for v in result]
        return [float(result)]
    if isinstance(values, Sequence):
        return [float(v) for v in values]
    return [float(values)]  # type: ignore[arg-type]


class YoloBeverageDetector:
    def __init__(
        self,
        config: Optional[BeverageDetectorConfig] = None,
        model: object | None = None,
    ) -> None:
        self.config = config or BeverageDetectorConfig()
        self._allowed_labels = {label.lower() for label in self.config.allowed_labels}
        if model is not None:
            self._model = model
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YoloBeverageDetector. "
                "Install with: pip install ultralytics"
            ) from exc
        self._model = YOLO(self.config.model_path)

    def _parse_results(self, results: Sequence[object]) -> List[BeverageDetection]:
        detections: List[BeverageDetection] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy_values = _to_list(getattr(boxes, "xyxy", []))
            conf_values = _to_list(getattr(boxes, "conf", []))
            cls_values = _to_list(getattr(boxes, "cls", []))

            # Expected flat xyxy list is 4*N entries.
            if len(xyxy_values) % 4 != 0:
                continue
            num_boxes = len(xyxy_values) // 4
            if len(conf_values) < num_boxes or len(cls_values) < num_boxes:
                continue

            for i in range(num_boxes):
                score = float(conf_values[i])
                if score < self.config.conf_min:
                    continue
                cls_idx = int(cls_values[i])
                raw_label = str(names.get(cls_idx, ""))
                label = _normalize_label(raw_label)
                if label is None or label.lower() not in self._allowed_labels:
                    continue
                x1, y1, x2, y2 = xyxy_values[i * 4 : (i + 1) * 4]
                detections.append(
                    BeverageDetection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        label=label,
                        score=score,
                    )
                )
        return detections

    def detect(self, frame: np.ndarray) -> List[BeverageDetection]:
        results = self._model(frame, verbose=False)
        if not isinstance(results, Sequence):
            results = [results]
        return self._parse_results(results)
