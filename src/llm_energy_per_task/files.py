"""File writes that cannot leave a half-written result behind.

A measurement costs GPU time to redo, so a path here only ever holds a complete
file: the new content lands beside it and is swapped in whole.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    write_atomic(path, json.dumps(value, indent=1) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    write_atomic(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
