from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from person_id_pi.beverage_types import BeerLabels, BeverageEvent, EspressoLabels

class BeverageStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: Dict[str, BeverageEvent] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists() or not self.path.read_text().strip():
            self._events = {}
            return
        data = json.loads(self.path.read_text())
        self._events = {
            entry["event_id"]: BeverageEvent.from_dict(entry) for entry in data
        }

    def save(self) -> None:
        payload = [event.to_dict() for event in self._events.values()]
        self.path.write_text(json.dumps(payload, indent=2))

    def list_events(self) -> List[BeverageEvent]:
        return sorted(self._events.values(), key=lambda e: e.timestamp_utc)

    def add_event(self, event: BeverageEvent) -> bool:
        if event.event_id in self._events:
            return False
        self._events[event.event_id] = event
        self.save()
        return True

    def total_beers_by_user(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for event in self._events.values():
            if event.beverage_label in BeerLabels:  
                summary[event.user_id] = summary.get(event.user_id, 0) + 1
        return summary
    
    def total_espressos_by_user(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for event in self._events.values():
            if event.beverage_label in EspressoLabels:  
                summary[event.user_id] = summary.get(event.user_id, 0) + 1
        return summary