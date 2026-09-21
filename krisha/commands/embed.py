"""`embed`: Phase B wrapper around embed_worker (guarded heavy imports)."""
from __future__ import annotations

from ..db import Database


def run(db: Database, batch_size: int = 16, limit: int | None = None) -> None:
    try:
        from ..embed_worker import run as _run
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Embedding deps missing. Install Phase B extras: "
            "pip install open-clip-torch torch pillow qdrant-client\n"
            f"({e})"
        )
    _run(db, batch_size=batch_size, limit=limit)
