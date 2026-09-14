"""Saved specification runs, so results can be shared by link and revisited."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, DateTime, Engine, Numeric, String, Table, Text, func, select

from app.db.postgres import get_engine, metadata

log = logging.getLogger(__name__)

spec_runs = Table(
    "spec_runs",
    metadata,
    Column("request_id", String(32), primary_key=True),
    Column("created_at", DateTime(timezone=True), server_default=func.current_timestamp(), index=True),
    Column("status", String(16), nullable=False),
    Column("finish_style", String(64)),
    Column("total_usd", Numeric(10, 2), nullable=False),
    Column("max_width_cm", Numeric(6, 2), nullable=False),
    Column("budget_usd", Numeric(10, 2), nullable=False),
    Column("had_image", String(5), nullable=False, server_default="false"),
    Column("payload", Text, nullable=False),  # full SpecResponse JSON (the photo itself is never stored)
)


def save_spec_run(response: dict[str, Any], engine: Engine | None = None) -> None:
    c = response["constraints"]
    row = {
        "request_id": response["request_id"],
        # Set here rather than by the database: SQLite's CURRENT_TIMESTAMP has 1 s resolution.
        "created_at": datetime.now(timezone.utc),
        "status": response["status"],
        "finish_style": response.get("finish_style_resolved"),
        "total_usd": response["total_usd"],
        "max_width_cm": c["max_width_cm"],
        "budget_usd": c["budget_usd"],
        "had_image": "true" if response.get("image_analysis") else "false",
        "payload": json.dumps(response, separators=(",", ":")),
    }
    with (engine or get_engine()).begin() as conn:
        conn.execute(spec_runs.insert().values(**row))


def get_spec_run(request_id: str, engine: Engine | None = None) -> dict[str, Any] | None:
    stmt = select(spec_runs.c.payload).where(spec_runs.c.request_id == request_id)
    with (engine or get_engine()).connect() as conn:
        payload = conn.execute(stmt).scalar_one_or_none()
    return json.loads(payload) if payload else None


def list_spec_runs(limit: int = 20, engine: Engine | None = None) -> list[dict[str, Any]]:
    t = spec_runs
    stmt = (
        select(
            t.c.request_id,
            t.c.created_at,
            t.c.status,
            t.c.finish_style,
            t.c.total_usd,
            t.c.max_width_cm,
            t.c.budget_usd,
            t.c.had_image,
        )
        .order_by(t.c.created_at.desc(), t.c.request_id)
        .limit(limit)
    )
    with (engine or get_engine()).connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(stmt)]
    for r in rows:
        for k in ("total_usd", "max_width_cm", "budget_usd"):
            r[k] = float(r[k])
        r["had_image"] = r["had_image"] == "true"
    return rows
