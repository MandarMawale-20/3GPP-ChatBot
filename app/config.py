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

        # Reject the legacy single-release schema (top-level `release`/`documents`)
        # up front: under the multi-release model a release must be addressed
        # explicitly, so a scalar `release:` can never unambiguously resolve.
        if "release" in self.corpus and "releases" not in self.corpus:
            raise ValueError(
                "configs/corpus.yaml uses the legacy single-release schema "
                "(top-level 'release'/'documents'). Multi-release requires the "
                "'releases' map. Migrate corpus.yaml before proceeding."
            )

        # target_release (env TARGET_RELEASE) is the *default* release for
        # single-release CLI operations; it must be one of the enabled releases
        # so an unscoped command never silently targets a disabled release.
        if self.settings.target_release not in self.enabled_releases:
            raise ValueError(
                f"TARGET_RELEASE={self.settings.target_release!r} is not an "
                f"enabled release. Enabled releases: {self.enabled_releases}."
            )

    @property
    def default_release(self) -> str:
        """Release used when a caller doesn't specify one (env TARGET_RELEASE)."""
        return self.settings.target_release

    @property
    def enabled_releases(self) -> list[str]:
        """Every release marked `enabled: true` in corpus.yaml, stable order."""
        releases = self.corpus.get("releases", {}) or {}
        return [r for r, cfg in releases.items() if cfg.get("enabled", True)]

    def release_config(self, release: str) -> dict:
        """The config block for one release; fail clearly if absent/disabled."""
        releases = self.corpus.get("releases", {}) or {}
        if release not in releases:
            raise ValueError(f"Release {release!r} is not configured in corpus.yaml.")
        cfg = releases[release]
        if not cfg.get("enabled", True):
            raise ValueError(f"Release {release!r} is disabled in corpus.yaml.")
        return cfg

    def allowed_documents(self, release: str | None = None) -> list[dict]:
        """Every allowlisted document, optionally scoped to one release.

        Each dict carries its spec_number, series, title, plus the release
        context (``release``/``release_number``) it belongs to. ``release=None``
        returns documents across all enabled releases (duplicates across
        releases are intentionally preserved — 24.501 legitimately exists in
        both Rel-17 and Rel-18 and must be indexed once per release).
        """
        docs: list[dict] = []
        targets = self.enabled_releases if release is None else [release]
        for r in targets:
            cfg = self.release_config(r)
            for group in cfg.get("documents", {}).values():
                for d in group:
                    entry = dict(d)  # don't mutate the loaded YAML
                    entry["release"] = r
                    entry["release_number"] = cfg["release_number"]
                    docs.append(entry)
        return docs

    @property
    def allowed_spec_numbers(self) -> list[str]:
        """Flat list of every spec_number across all enabled releases."""
        return [d["spec_number"] for d in self.allowed_documents()]


@lru_cache
def get_settings() -> AppConfig:
    """Process-wide singleton. Cached so YAML/env are parsed only once."""
    return AppConfig()
