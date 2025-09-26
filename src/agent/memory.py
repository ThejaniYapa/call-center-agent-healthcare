from __future__ import annotations

from pathlib import Path
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver


def get_checkpointer(db_path: str | Path = ".checkpoints/state.db") -> SqliteSaver:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_file(str(path))

