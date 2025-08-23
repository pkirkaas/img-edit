"""
Tests for the Hugging Face client in [src/lib/hf_client.py](src/lib/hf_client.py).

Focus:
- Successful API calls with mocked httpx responses
- Error handling for HTTP errors and network issues
- Retry mechanism behavior
- Authentication header presence and payload preparation

All external network calls are mocked to avoid real requests.

Note: Python syntax validated via ast prior to submission.
"""
from __future__ import annotations

import base64
import io
import json
import importlib
import sys
from typing import Any, Dict, Generator

import httpx
import pytest
from PIL import Image


def _make_image_bytes(fmt: str = "JPEG", size: tuple[int, int] = (6, 4), color: str = "red") -> bytes:
    """Create a tiny in-memory image and return its raw bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture()
def hf_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """Import lib.config and lib.hf_client after setting env; return (cfg_mod, hf_mod)."""
    monkeypatch.setenv("HF_API_TOKEN", "test-token")
    # Reload modules to ensure module-level side effects see the env var
    sys.modules.pop("lib.config", None)
    cfg = importlib.import_module("lib.config")
    sys.modules.pop("lib.hf_client", None)
    hf_mod = importlib.import_module("lib.hf_client")
    return cfg, hf_mod

@pytest.fixture()
def settings_env(hf_modules: tuple[Any, Any]) -> Any:
    """Provide a fresh Settings with a known token and short timeout."""
    cfg, _ = hf_modules
    # Instantiate Settings explicitly to avoid relying on global instance
    return cfg.Settings(HF_API_TOKEN="test-token", IMG_EDIT_TIMEOUT_SECONDS=5)


@pytest.fixture()
def client(hf_modules: tuple[Any, Any], settings_env: Any) -> Generator[Any, None, None]:
    """Yield an HFClient with small retry delay; ensure cleanup."""
    _, hf_mod = hf_modules
    c = hf_mod.HFClient(settings_env, max_retries=2, retry_delay=0.0)
    try:
        yield c
    finally:
        c.close()


def test_auth_headers_included(client: HFClient) -> None:
    # The underlying httpx.Client should have correct headers
    headers = client.client.headers
    assert headers.get("Authorization") == "Bearer test-token"
    assert headers.get("Content-Type") == "application/json"


def test_prepare_payload_encodes_image_and_includes_params(client: HFClient) -> None:
    img_bytes = _make_image_bytes()
    payload = client._prepare_payload(  # noqa: SLF001 - intentionally testing a private helper
        image_bytes=img_bytes,
        prompt="Replace the sky with a sunset",
        guidance_scale=7.5,
        strength=0.8,
        seed=42,
        mask="b64-mask-string",
        some_extra="value",
    )
    assert "inputs" in payload
    inputs: Dict[str, Any] = payload["inputs"]
    assert inputs["prompt"].startswith("Replace the sky")
    # Base64 round-trip check
    decoded = base64.b64decode(inputs["image"])
    assert decoded == img_bytes
    # Optional/extra fields included
    assert inputs["seed"] == 42
    assert inputs["mask"] == "b64-mask-string"
    assert inputs["some_extra"] == "value"


def test_successful_api_call_returns_pil_image(monkeypatch: pytest.MonkeyPatch, client: HFClient) -> None:
    # Prepare mocked successful JSON response with image
    img_bytes = _make_image_bytes(size=(5, 5), color="green")
    resp_json = {"image": base64.b64encode(img_bytes).decode("utf-8")}

    def fake_post(url: str, json: Dict[str, Any]) -> httpx.Response:  # type: ignore[override]
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            content=jsonlib.dumps(resp_json).encode("utf-8"),
        )

    # Bind a minimal json lib alias to avoid shadowing
    jsonlib = json
    monkeypatch.setattr(client.client, "post", fake_post)  # replace instance method

    result = client.inference_request(image_bytes=img_bytes, prompt="x")
    assert isinstance(result, Image.Image)
    assert result.size == (5, 5)


def test_http_error_raises_hfapierror(monkeypatch: pytest.MonkeyPatch, hf_modules: tuple[Any, Any], settings_env: Any) -> None:
    _, hf_mod = hf_modules
    # Single-attempt client to exercise final failure (no retries)
    c = hf_mod.HFClient(settings_env, max_retries=0, retry_delay=0.0)

    def fake_post(url: str, json: Dict[str, Any]) -> httpx.Response:  # type: ignore[override]
        # Simulate 500 response; raise_for_status() will raise HTTPStatusError in client code
        return httpx.Response(
            500,
            request=httpx.Request("POST", url),
            content=b'{"error":"server"}',
        )

    try:
        monkeypatch.setattr(c.client, "post", fake_post)
        with pytest.raises(hf_mod.HFAPIError) as ei:
            c.inference_request(image_bytes=_make_image_bytes(), prompt="boom")
        # Optionally assert message contains HTTP code
        assert "API request failed" in str(ei.value)
    finally:
        c.close()


def test_network_error_raises_hfnetworkerror(monkeypatch: pytest.MonkeyPatch, hf_modules: tuple[Any, Any], settings_env: Any) -> None:
    _, hf_mod = hf_modules
    c = hf_mod.HFClient(settings_env, max_retries=1, retry_delay=0.0)

    def fake_post(url: str, json: Dict[str, Any]) -> httpx.Response:  # type: ignore[override]
        raise httpx.RequestError("connection lost", request=httpx.Request("POST", url))

    try:
        monkeypatch.setattr(c.client, "post", fake_post)
        with pytest.raises(hf_mod.HFNetworkError):
            c.inference_request(image_bytes=_make_image_bytes(), prompt="boom")
    finally:
        c.close()


def test_timeout_error_raises_hfnetworkerror(monkeypatch: pytest.MonkeyPatch, hf_modules: tuple[Any, Any], settings_env: Any) -> None:
    _, hf_mod = hf_modules
    c = hf_mod.HFClient(settings_env, max_retries=0, retry_delay=0.0)

    def fake_post(url: str, json: Dict[str, Any]) -> httpx.Response:  # type: ignore[override]
        raise httpx.TimeoutException("timed out")

    try:
        monkeypatch.setattr(c.client, "post", fake_post)
        with pytest.raises(hf_mod.HFNetworkError):
            c.inference_request(image_bytes=_make_image_bytes(), prompt="timeout")
    finally:
        c.close()


def test_retry_then_success(monkeypatch: pytest.MonkeyPatch, client: HFClient) -> None:
    calls = {"n": 0}
    img_bytes = _make_image_bytes(size=(3, 3))
    ok_json = {"image": base64.b64encode(img_bytes).decode("utf-8")}

    def fake_post(url: str, json: Dict[str, Any]) -> httpx.Response:  # type: ignore[override]
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(503, request=httpx.Request("POST", url), content=b'{"error":"busy"}')
        return httpx.Response(200, request=httpx.Request("POST", url), content=json.dumps(ok_json).encode("utf-8"))

    monkeypatch.setattr(client.client, "post", fake_post)
    img = client.inference_request(image_bytes=img_bytes, prompt="try again")
    assert isinstance(img, Image.Image)
    assert calls["n"] == 2


def test_parse_response_errors(client: Any, hf_modules: tuple[Any, Any]) -> None:
    _, hf_mod = hf_modules
    # Missing 'image' key
    with pytest.raises(hf_mod.HFAPIError):
        client._parse_response({})  # noqa: SLF001
    # Invalid Base64 content
    with pytest.raises(hf_mod.HFAPIError):
        client._parse_response({"image": "%%%INVALID%%%"})  # noqa: SLF001