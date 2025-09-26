from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

try:
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    MemorySaver = None  # type: ignore[assignment]

try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    SqliteSaver = None  # type: ignore[assignment]

CheckpointSaver: TypeAlias = "SqliteSaver | MemorySaver | None"


def get_checkpointer(db_path: str | Path = ".checkpoints/state.db") -> CheckpointSaver:
    if SqliteSaver is not None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_file(str(path))
    if MemorySaver is not None:
        return MemorySaver()
    return None
