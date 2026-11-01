"""JSONL trace logger for the RepoIntel Investigator."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


class Tracer:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, step: int, thought: str, action: str,
            args: dict, observation: dict) -> None:
        record = {
            "step": step,
            "ts": datetime.now(timezone.utc).isoformat(),
            "thought": thought,
            "action": action,
            "args": args,
            "observation": observation,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "investigation"
