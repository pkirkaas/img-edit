"""
Tests for configuration management in [src/lib/config.py](src/lib/config.py).

Covers:
- Loading from environment variables
- Token preference HF_TOKEN over HF_API_TOKEN
- Default values and explicit overrides
- Endpoint validation rules

Note: Python syntax validated via ast prior to submission.
"""
from __future__ import annotations

import importlib
import sys
from typing import Any, Dict

import pytest


def _reload_config_with_env(env: Dict[str, Any]) -> Any:
    """
    Helper to reload lib.config with a controlled environment.

    Ensures tests are isolated from global import side effects caused by the
    module-level 'settings = Settings()' in [src/lib/config.py](src/lib/config.py).
    """
    # Apply environment variables
    from os import environ

    for k, v in env.items():
        environ[k] = str(v)
    # Drop module to force re-evaluation of globals (Settings(), validators, etc.)
    sys.modules.pop("lib.config", None)
    cfg = importlib.import_module("lib.config")
    return cfg


def test_settings_loads_from_env_and_token_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: only HF_TOKEN set (preferred)
    monkeypatch.setenv("HF_TOKEN", "abc123TOKEN")
    # Act: import/reload config
    cfg = _reload_config_with_env({"HF_TOKEN": "abc123TOKEN"})
    # Assert: token preference and defaults
    assert cfg.settings.get_token() == "abc123TOKEN"
    assert cfg.settings.IMG_EDIT_PROVIDER == "fal-ai"
    assert cfg.settings.IMG_EDIT_MODEL == "Qwen/Qwen-Image-Edit"
    # Endpoint is optional and defaults to None now
    assert cfg.settings.HF_INFERENCE_ENDPOINT is None
    assert isinstance(cfg.settings.IMG_EDIT_TIMEOUT_SECONDS, int) and cfg.settings.IMG_EDIT_TIMEOUT_SECONDS == 120
    assert isinstance(cfg.settings.IMG_EDIT_DEFAULT_GUIDANCE, float) and cfg.settings.IMG_EDIT_DEFAULT_GUIDANCE == 7.5
    assert isinstance(cfg.settings.IMG_EDIT_DEFAULT_STRENGTH, float) and cfg.settings.IMG_EDIT_DEFAULT_STRENGTH == 0.8
    # Optional seed default is None
    assert cfg.settings.IMG_EDIT_SEED is None


def test_token_preference_hf_token_over_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _reload_config_with_env({"HF_TOKEN": "preferred", "HF_API_TOKEN": "fallback"})
    assert cfg.settings.get_token() == "preferred"
    # Clearing HF_TOKEN falls back to HF_API_TOKEN
    cfg = _reload_config_with_env({"HF_API_TOKEN": "fallback-only"})
    assert cfg.settings.get_token() == "fallback-only"


def test_endpoint_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _reload_config_with_env({})
    # Invalid scheme should raise
    with pytest.raises(ValueError):
        cfg.Settings(HF_INFERENCE_ENDPOINT="ftp://not-allowed")
    # Valid http/https pass through and are normalized (trailing slash trimmed)
    s = cfg.Settings(HF_INFERENCE_ENDPOINT="https://example.com/path/")
    assert s.HF_INFERENCE_ENDPOINT == "https://example.com/path"


def test_explicit_overrides_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _reload_config_with_env({})
    s = cfg.Settings(
        HF_API_TOKEN="valid3",
        IMG_EDIT_TIMEOUT_SECONDS=5,
        IMG_EDIT_PROVIDER="my-provider",
        IMG_EDIT_MODEL="Org/Model",
    )
    assert s.IMG_EDIT_TIMEOUT_SECONDS == 5
    assert s.IMG_EDIT_PROVIDER == "my-provider"
    assert s.IMG_EDIT_MODEL == "Org/Model"
    # Ensure defaults unchanged for others
    assert s.IMG_EDIT_DEFAULT_GUIDANCE == 7.5
    assert s.IMG_EDIT_DEFAULT_STRENGTH == 0.8