"""Application settings loaded from environment variables / `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RoomSpec AI"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # Relational store. Any SQLAlchemy URL works: Neon / Supabase in production,
    # a local SQLite file for zero-config development.
    database_url: str = f"sqlite:///{REPO_ROOT / 'data' / 'roomspec.db'}"

    # Vector store. Leave `qdrant_url` empty to run Qdrant in embedded in-memory mode.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "cabinet_modules"

    # Embeddings. `auto` uses CLIP when torch/transformers are importable, else hashing.
    embedding_backend: Literal["auto", "clip", "hash"] = "auto"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512

    # LLM for installation notes. `none` renders a deterministic template instead.
    llm_provider: Literal["auto", "groq", "gemini", "none"] = "auto"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    llm_timeout_s: float = 20.0

    # Retrieval / upload limits.
    vector_top_k: int = Field(default=40, ge=1, le=200)
    max_upload_mb: float = 8.0
    auto_seed: bool = True
    seed_file: Path = REPO_ROOT / "data" / "seed_modules.json"
    frontend_dir: Path = REPO_ROOT / "frontend"


@lru_cache
def get_settings() -> Settings:
    return Settings()
