"""Centralized configuration.

Two layers, deliberately kept separate:

1. `.env` — secrets and deployment-specific values (API keys, URLs, ports)
   plus retrieval/evidence-gate tuning, loaded via pydantic-settings.
2. `configs/settings.yaml` + `configs/corpus.yaml` — non-secret chunking
   values and the release/document allowlist, checked into version control.

Every other module should import `get_settings()` rather than reading
os.environ or YAML directly, so there is one source of truth for
configuration at runtime.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


class Settings(BaseSettings):
    """Secrets and per-deployment values, sourced from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: Literal["gemini", "openai", "local"] = Field(default="gemini")
    llm_model: str = Field(default="gemini-2.0-flash")
    gemini_api_key: str = Field(default="")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")
    qdrant_collection: str = Field(default="3gpp_standards")

    # Embeddings
    embedding_model: str = Field(default="BAAI/bge-m3")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Chunks embedded per forward pass during ingestion. Larger batches are
    # faster on GPU but use more VRAM; keep modest on CPU.
    embedding_batch_size: int = Field(default=16)

    # Corpus control — frozen to Rel-18. configs/corpus.yaml is the
    # detailed allowlist.
    target_release: str = Field(default="Rel-18")

    # Retrieval tuning (overridable via environment variables)
    dense_top_k: int = Field(default=20)
    sparse_top_k: int = Field(default=20)
    fused_top_k: int = Field(default=30)
    rerank_top_k: int = Field(default=8)
    evidence_score_threshold: float = Field(default=0.35)

    # App
    app_env: Literal["development", "production", "test"] = Field(default="development")
    log_level: str = Field(default="INFO")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        # Fail loudly: a missing corpus.yaml means we don't know what is
        # safe to index.
        raise FileNotFoundError(f"Required config file missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class AppConfig:
    """Aggregates env Settings with the non-secret YAML configuration."""

    def __init__(self) -> None:
        self.settings = Settings()
        self.corpus: dict = _load_yaml(CONFIGS_DIR / "corpus.yaml")
        self.tuning: dict = _load_yaml(CONFIGS_DIR / "settings.yaml")

        # Guard: the env-level target_release must match the corpus
        # allowlist's release; fail fast instead of silently mixing
        # releases.
        if self.settings.target_release != self.corpus.get("release"):
            raise ValueError(
                "target_release mismatch: "
                f"env={self.settings.target_release!r} "
                f"corpus.yaml={self.corpus.get('release')!r}. "
                "Cross-release contamination is not allowed."
            )

    @property
    def allowed_spec_numbers(self) -> list[str]:
        """Flat list of every spec_number this deployment may ingest."""
        docs = self.corpus.get("documents", {})
        return [d["spec_number"] for group in docs.values() for d in group]


@lru_cache
def get_settings() -> AppConfig:
    """Process-wide singleton. Cached so YAML/env are parsed only once."""
    return AppConfig()
