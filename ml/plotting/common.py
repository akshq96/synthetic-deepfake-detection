"""Shared plotting setup. Every plot function in ml/plotting/ saves both a
.png (paper figure) and the underlying data as .csv/.json (so tables/figures
can be regenerated without rerunning inference) — see the project plan's
Evaluation module design.
"""

from __future__ import annotations

from pathlib import Path


def use_non_interactive_backend() -> None:
    """Must be called before importing pyplot in any headless context (CI,
    a server, a test suite) — matplotlib's default backend can require a
    display otherwise.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
