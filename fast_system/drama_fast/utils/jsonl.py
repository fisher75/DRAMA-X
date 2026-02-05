"""Small JSONL helper functions used across DRAMA-X fast_system.

We keep this minimal and dependency-free on purpose.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a .jsonl file into a list of python dicts."""
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def dump_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    """Write iterable of dicts to a .jsonl file."""
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
