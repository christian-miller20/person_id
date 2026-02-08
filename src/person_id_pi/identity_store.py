from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class UserTemplates:
    user_id: str
    templates: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {"user_id": self.user_id, "templates": self.templates}

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "UserTemplates":
        return cls(
            user_id=str(payload["user_id"]),
            templates=[list(vec) for vec in payload.get("templates", [])],
        )


class IdentityStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._users: Dict[str, UserTemplates] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists() or not self.path.read_text().strip():
            self._users = {}
            return
        data = json.loads(self.path.read_text())
        self._users = {
            entry["user_id"]: UserTemplates.from_dict(entry) for entry in data
        }

    def save(self) -> None:
        payload = [user.to_dict() for user in self._users.values()]
        self.path.write_text(json.dumps(payload, indent=2))

    def list_users(self) -> List[str]:
        return sorted(self._users.keys())

    def has_user(self, user_id: str) -> bool:
        return user_id in self._users

    def get_templates(self, user_id: str) -> List[np.ndarray]:
        user = self._users.get(user_id)
        if user is None:
            return []
        return [np.asarray(vec, dtype=np.float32) for vec in user.templates]

    def add_user(self, user_id: str) -> None:
        if user_id not in self._users:
            self._users[user_id] = UserTemplates(user_id=user_id)
            self.save()

    def delete_user(self, user_id: str) -> bool:
        if user_id not in self._users:
            return False
        del self._users[user_id]
        self.save()
        return True

    def rename_user(self, cur_user_id: str, new_user_id: str) -> bool:
        if cur_user_id not in self._users or new_user_id in self._users:
            return False
        self._users[new_user_id] = self._users[cur_user_id]
        self._users[new_user_id].user_id = new_user_id
        del self._users[cur_user_id]
        self.save()
        return True

    def add_template(self, user_id: str, embedding: np.ndarray) -> None:
        if user_id not in self._users:
            self._users[user_id] = UserTemplates(user_id=user_id)
        user = self._users[user_id]
        user.templates.append(embedding.astype(np.float32).tolist())
        self.save()

    def replace_templates(self, user_id: str, templates: List[np.ndarray]) -> None:
        if user_id not in self._users:
            self._users[user_id] = UserTemplates(user_id=user_id)
        self._users[user_id].templates = [
            t.astype(np.float32).tolist() for t in templates
        ]
        self.save()

    def rename_user(self, current_user_id: str, new_user_id: str) -> bool:
        if current_user_id not in self._users:
            return False
        if current_user_id == new_user_id:
            return True
        if new_user_id in self._users:
            return False
        record = self._users.pop(current_user_id)
        record.user_id = new_user_id
        self._users[new_user_id] = record
        self.save()
        return True

    def generate_new_user_id(self) -> str:
        pattern = re.compile(r"^user_(\d+)$")
        max_idx = -1
        for user_id in self._users.keys():
            match = pattern.match(user_id)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        return f"user_{max_idx + 1:04d}"
