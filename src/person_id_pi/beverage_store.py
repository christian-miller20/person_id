from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List

from .beverage_types import BeerLabels, BeverageEvent, EspressoLabels


class BeverageStore:
    """JSON-backed event store with idempotent inserts by event_id."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: Dict[str, BeverageEvent] = {}
        self._load()

    def _load(self) -> None:
        """Load existing events if the file exists and is non-empty."""
        if not self.path.exists():
            self._events = {}
            return
        raw = self.path.read_text(encoding="utf-8").strip()
        if not raw:
            self._events = {}
            return

        data = json.loads(raw)
        self._events = {
            entry["event_id"]: BeverageEvent.from_dict(entry) for entry in data
        }

    def save(self) -> None:
        """Persist all events to disk."""
        payload = [event.to_dict() for event in self.list_events()]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_events(self) -> List[BeverageEvent]:
        return sorted(self._events.values(), key=lambda e: e.timestamp_utc)

    def add_event(self, event: BeverageEvent) -> bool:
        """Insert once by deterministic event_id; returns False on duplicate."""
        if event.event_id in self._events:
            return False
        self._events[event.event_id] = event
        self.save()
        return True

    def total_beers_by_user(self) -> Dict[str, int]:
        """All-time beer-like totals grouped by user_id."""
        return self._count_by_user_for_labels(BeerLabels)

    def total_espressos_by_user(self) -> Dict[str, int]:
        """All-time espresso totals grouped by user_id."""
        return self._count_by_user_for_labels(EspressoLabels)

    def _count_by_user_for_labels(self, labels: Iterable[str]) -> Dict[str, int]:
        """Shared aggregation helper for per-user beverage totals."""
        label_set = set(labels)
        summary: Dict[str, int] = {}
        for event in self._events.values():
            if event.beverage_label in label_set:
                summary[event.user_id] = summary.get(event.user_id, 0) + 1
        return summary
