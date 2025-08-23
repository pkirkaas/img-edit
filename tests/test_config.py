"""
Tests for configuration management in [src/lib/config.py](src/lib/config.py).

Covers:
- Loading from environment variables
- Validation errors for missing/invalid variables
- Default values and explicit overrides

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

    for k in list(env.keys()):
        environ[k] = str(env[k])
    # Drop module to force re-evaluation of globals (Settings(), validators, etc.)
    sys.modules.pop("lib.config", None)
    cfg = importlib.import_module("lib.config")
    return cfg


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: set required env before import
    monkeypatch.setenv("HF_API_TOKEN", "abc123TOKEN")
    # Act: import/reload config
    cfg = _reload_config_with_env({"HF_API_TOKEN": "abc123TOKEN"})
    # Assert: global settings picks up env
    assert cfg.settings.HF_API_TOKEN == "abc123TOKEN"
    # Defaults are set correctly
    assert (
        cfg.settings.HF_INFERENCE_ENDPOINT
        == "https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit"
    )
    assert isinstance(cfg.settings.IMG_EDIT_TIMEOUT_SECONDS, int) and cfg.settings.IMG_EDIT_TIMEOUT_SECONDS == 120
    assert isinstance(cfg.settings.IMG_EDIT_DEFAULT_GUIDANCE, float) and cfg.settings.IMG_EDIT_DEFAULT_GUIDANCE == 7.5
    assert isinstance(cfg.settings.IMG_EDIT_DEFAULT_STRENGTH, float) and cfg.settings.IMG_EDIT_DEFAULT_STRENGTH == 0.8
    # Optional seed default is None
    assert cfg.settings.IMG_EDIT_SEED is None


def test_validation_error_missing_token_on_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure module imports successfully (needs a token)
    monkeypatch.setenv("HF_API_TOKEN", "valid")
    cfg = _reload_config_with_env({"HF_API_TOKEN": "valid"})
    # Constructing Settings with an empty token should raise
    with pytest.raises(ValueError):
        cfg.Settings(HF_API_TOKEN="")


def test_endpoint_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_API_TOKEN", "valid2")
    cfg = _reload_config_with_env({"HF_API_TOKEN": "valid2"})
    # Invalid scheme should raise
    with pytest.raises(ValueError):
        cfg.Settings(HF_API_TOKEN="valid2", HF_INFERENCE_ENDPOINT="ftp://not-allowed")


def test_explicit_overrides_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_API_TOKEN", "valid3")
    cfg = _reload_config_with_env({"HF_API_TOKEN": "valid3"})
    s = cfg.Settings(HF_API_TOKEN="valid3", IMG_EDIT_TIMEOUT_SECONDS=5)
    assert s.IMG_EDIT_TIMEOUT_SECONDS == 5
    # Ensure defaults unchanged for others
    assert s.IMG_EDIT_DEFAULT_GUIDANCE == 7.5
    assert s.IMG_EDIT_DEFAULT_STRENGTH == 0.8