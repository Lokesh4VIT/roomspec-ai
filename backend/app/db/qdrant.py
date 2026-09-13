"""Qdrant vector store: Qdrant Cloud when `QDRANT_URL` is set, embedded in-memory otherwise."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from app.core.config import get_settings

log = logging.getLogger(__name__)

_POINT_NS = uuid.UUID("6f1c5d8e-6a1b-4c3e-9b0e-2d9b1e7f4a10")
PAYLOAD_INDEXES = ("part_id", "category", "finish_style")


@dataclass(frozen=True)
class VectorHit:
    part_id: str
    score: float
    payload: dict[str, Any]


def point_id(part_id: str) -> str:
    return str(uuid.uuid5(_POINT_NS, part_id))


@lru_cache
def get_qdrant() -> QdrantClient:
    s = get_settings()
    if s.qdrant_url:
        log.info("Connecting to Qdrant at %s", s.qdrant_url)
        return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None, timeout=10)
    log.info("QDRANT_URL not set; using embedded in-memory Qdrant")
    return QdrantClient(location=":memory:")


def collection_name() -> str:
    """Collection namespaced by embedding backend.

    Hash and CLIP vectors are both 512-d but live in unrelated spaces; a shared collection
    would silently return garbage after switching backends, so each backend gets its own.
    """
    from app.services.embedding_engine import get_embedder

    return f"{get_settings().qdrant_collection}_{get_embedder().name}"


DENSE = "dense"
LEXICAL = "lexical"


def _collection_matches(client: QdrantClient, name: str, dim: int) -> bool:
    params = client.get_collection(name).config.params
    vectors = params.vectors
    dense_ok = isinstance(vectors, dict) and DENSE in vectors and vectors[DENSE].size == dim
    sparse_ok = bool(params.sparse_vectors) and LEXICAL in params.sparse_vectors
    return dense_ok and sparse_ok


def ensure_collection(dim: int, client: QdrantClient | None = None, recreate: bool = False) -> None:
    """Named dense (CLIP/hash, cosine HNSW) + sparse lexical (BM25-style, server-side IDF) vectors."""
    client = client or get_qdrant()
    name = collection_name()
    exists = client.collection_exists(name)
    if exists and not recreate:
        if _collection_matches(client, name, dim):
            return
        log.warning("Collection %s has an outdated vector config; recreating", name)
    if exists:
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={LEXICAL: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
    )
    if get_settings().qdrant_url:  # payload indexes are a no-op in embedded mode
        for field in PAYLOAD_INDEXES:
            client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)


def collection_count(client: QdrantClient | None = None) -> int:
    client = client or get_qdrant()
    name = collection_name()
    if not client.collection_exists(name):
        return 0
    return client.count(name, exact=True).count


def upsert_vectors(
    part_ids: Sequence[str],
    vectors: np.ndarray,
    sparse: Sequence[tuple[list[int], list[float]]],
    payloads: Sequence[dict[str, Any]],
    client: QdrantClient | None = None,
) -> None:
    client = client or get_qdrant()
    points = [
        models.PointStruct(
            id=point_id(pid),
            vector={DENSE: vec.tolist(), LEXICAL: models.SparseVector(indices=idx, values=val)},
            payload={"part_id": pid, **pl},
        )
        for pid, vec, (idx, val), pl in zip(part_ids, vectors, sparse, payloads, strict=True)
    ]
    client.upsert(collection_name(), points=points, wait=True)


def search(
    vector: np.ndarray | tuple[list[int], list[float]],
    limit: int,
    categories: Sequence[str] | None = None,
    client: QdrantClient | None = None,
) -> list[VectorHit]:
    """Dense search for an ndarray, sparse lexical search for an (indices, values) pair."""
    client = client or get_qdrant()
    query_filter = None
    if categories:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="category", match=models.MatchAny(any=list(categories)))]
        )
    if isinstance(vector, tuple):
        if not vector[0]:
            return []
        query, using = models.SparseVector(indices=vector[0], values=vector[1]), LEXICAL
    else:
        query, using = vector.tolist(), DENSE
    res = client.query_points(
        collection_name(),
        query=query,
        using=using,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )
    return [VectorHit(p.payload["part_id"], float(p.score), p.payload) for p in res.points]
