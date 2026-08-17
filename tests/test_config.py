"""Tests for app.config.

Focus: the settings/corpus YAML load correctly, and the release-consistency
guard actually fires — that guard is the main safety net preventing the
downloader/validator from ever operating outside Rel-18.
"""

from __future__ import annotations

import pytest

from app.config import AppConfig, get_settings


def test_get_settings_returns_singleton() -> None:
    a = get_settings()
    b = get_settings()
    assert a is b


def test_corpus_release_matches_env_default() -> None:
    config = get_settings()
    assert config.settings.target_release == config.corpus["release"]


def test_allowed_spec_numbers_includes_core_documents() -> None:
    config = get_settings()
    allowed = config.allowed_spec_numbers
    for spec in ("23.501", "23.502", "24.501"):
        assert spec in allowed


def test_allowed_spec_numbers_includes_extended_documents() -> None:
    config = get_settings()
    allowed = config.allowed_spec_numbers
    for spec in ("38.300", "38.331", "33.501", "29.244", "29.500"):
        assert spec in allowed


def test_release_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Changing TARGET_RELEASE without updating corpus.yaml must fail
    # loudly, not silently ingest a different release.
    monkeypatch.setenv("TARGET_RELEASE", "Rel-19")
    with pytest.raises(ValueError, match="target_release mismatch"):
        AppConfig()


def test_missing_corpus_file_raises(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.config as config_module

    monkeypatch.setattr(config_module, "CONFIGS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        AppConfig()
