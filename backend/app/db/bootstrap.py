"""Create schema, seed the catalog and build the vector index."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import get_settings
from app.db import postgres, qdrant, spec_store  # noqa: F401  (spec_store registers its table)

log = logging.getLogger(__name__)


def load_seed(path: Path | None = None) -> list[dict]:
    return json.loads((path or get_settings().seed_file).read_text())


def bootstrap(force_reseed: bool = False, force_reindex: bool = False) -> dict[str, int]:
    from app.services.embedding_engine import get_embedder
    from app.services.hybrid_retriever import index_catalog

    postgres.init_schema()
    rows = postgres.count_modules()
    seeded = 0
    if force_reseed or rows == 0:
        seeded = postgres.upsert_modules(load_seed())
        rows = postgres.count_modules()
        log.info("Seeded %s catalog modules", seeded)

    indexed = 0
    embedder = get_embedder()
    qdrant.ensure_collection(embedder.dim)
    if force_reindex or seeded or qdrant.collection_count() != rows:
        indexed = index_catalog(embedder=embedder)
        log.info("Indexed %s modules into Qdrant with %s embeddings", indexed, embedder.name)
    return {"catalog_rows": rows, "seeded": seeded, "indexed": indexed}
