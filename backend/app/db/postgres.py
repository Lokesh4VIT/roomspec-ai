"""Relational catalog store (PostgreSQL on Neon/Supabase, SQLite locally).

This layer owns every hard physical constraint: widths, depths, prices and stock.
Nothing downstream is allowed to invent these values.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Engine,
    Index,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    and_,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.core.config import get_settings

metadata = MetaData()

cabinet_modules = Table(
    "cabinet_modules",
    metadata,
    Column("part_id", String(64), primary_key=True),
    Column("part_name", String(255), nullable=False),
    Column("category", String(64), nullable=False),
    Column("finish_style", String(64), nullable=False),
    Column("material", String(128), nullable=False, server_default=""),
    Column("price_usd", Numeric(10, 2), nullable=False),
    Column("width_cm", Numeric(6, 2), nullable=False),
    Column("height_cm", Numeric(6, 2), nullable=False),
    Column("depth_cm", Numeric(6, 2), nullable=False),
    Column("door_clearance_cm", Numeric(6, 2), nullable=False, server_default="0"),
    Column("in_stock", Boolean, default=True),
    Column("description", Text, nullable=False, server_default=""),
    Column("image_url", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.current_timestamp()),
)

Index(
    "idx_modules_specs",
    cabinet_modules.c.category,
    cabinet_modules.c.width_cm,
    cabinet_modules.c.finish_style,
    cabinet_modules.c.in_stock,
)

_NUMERIC = ("price_usd", "width_cm", "height_cm", "depth_cm", "door_clearance_cm")


def _normalize_url(url: str) -> str:
    # Neon/Supabase hand out `postgres://` / `postgresql://`; use the psycopg3 driver.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    url = _normalize_url(url or get_settings().database_url)
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def init_schema(engine: Engine | None = None) -> None:
    metadata.create_all(engine or get_engine())


def _row_to_dict(row: Any) -> dict[str, Any]:
    d = dict(row._mapping)
    for key in _NUMERIC:
        if d.get(key) is not None:
            d[key] = float(d[key])
    d.pop("created_at", None)
    return d


def upsert_modules(rows: Iterable[dict[str, Any]], engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    rows = list(rows)
    if not rows:
        return 0
    insert = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
    stmt = insert(cabinet_modules).values(rows)
    update_cols = {c: stmt.excluded[c] for c in rows[0] if c != "part_id"}
    stmt = stmt.on_conflict_do_update(index_elements=["part_id"], set_=update_cols)
    with engine.begin() as conn:
        conn.execute(stmt)
    return len(rows)


def update_module(part_id: str, values: dict[str, Any], engine: Engine | None = None) -> dict[str, Any] | None:
    if values:
        stmt = cabinet_modules.update().where(cabinet_modules.c.part_id == part_id).values(**values)
        with (engine or get_engine()).begin() as conn:
            if conn.execute(stmt).rowcount == 0:
                return None
    return get_modules_by_ids([part_id], engine=engine).get(part_id)


def count_modules(engine: Engine | None = None) -> int:
    with (engine or get_engine()).connect() as conn:
        return conn.execute(select(func.count()).select_from(cabinet_modules)).scalar_one()


def get_modules_by_ids(part_ids: Sequence[str], engine: Engine | None = None) -> dict[str, dict]:
    if not part_ids:
        return {}
    stmt = select(cabinet_modules).where(cabinet_modules.c.part_id.in_(list(part_ids)))
    with (engine or get_engine()).connect() as conn:
        return {r.part_id: _row_to_dict(r) for r in conn.execute(stmt)}


def list_all_modules(engine: Engine | None = None) -> list[dict[str, Any]]:
    stmt = select(cabinet_modules).order_by(cabinet_modules.c.part_id)
    with (engine or get_engine()).connect() as conn:
        return [_row_to_dict(r) for r in conn.execute(stmt)]


def filter_modules(
    *,
    part_ids: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    finish_style: str | None = None,
    max_width_cm: float | None = None,
    max_depth_cm: float | None = None,
    max_height_cm: float | None = None,
    max_price_usd: float | None = None,
    in_stock_only: bool = True,
    limit: int | None = None,
    engine: Engine | None = None,
) -> list[dict[str, Any]]:
    """Hard-constraint SQL filter. Every returned row physically satisfies the bounds."""
    t = cabinet_modules
    clauses = []
    if part_ids is not None:
        if not part_ids:
            return []
        clauses.append(t.c.part_id.in_(list(part_ids)))
    if categories:
        clauses.append(t.c.category.in_(list(categories)))
    if finish_style:
        clauses.append(func.lower(t.c.finish_style) == finish_style.strip().lower())
    if max_width_cm is not None:
        clauses.append(t.c.width_cm <= max_width_cm)
    if max_depth_cm is not None:
        clauses.append(t.c.depth_cm <= max_depth_cm)
    if max_height_cm is not None:
        clauses.append(t.c.height_cm <= max_height_cm)
    if max_price_usd is not None:
        clauses.append(t.c.price_usd <= max_price_usd)
    if in_stock_only:
        clauses.append(t.c.in_stock.is_(True))

    stmt = select(t).order_by(t.c.category, t.c.width_cm, t.c.part_id)
    if clauses:
        stmt = stmt.where(and_(*clauses))
    if limit:
        stmt = stmt.limit(limit)
    with (engine or get_engine()).connect() as conn:
        return [_row_to_dict(r) for r in conn.execute(stmt)]


def distinct_values(column: str, exclude_categories: Sequence[str] = (), engine: Engine | None = None) -> list[str]:
    col = cabinet_modules.c[column]
    stmt = select(col).distinct().order_by(col)
    if exclude_categories:
        stmt = stmt.where(cabinet_modules.c.category.not_in(list(exclude_categories)))
    with (engine or get_engine()).connect() as conn:
        return [r[0] for r in conn.execute(stmt)]
