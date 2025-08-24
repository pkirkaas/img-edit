"""
Tests for HFImageEditClient in [src/lib/hf_client.py](src/lib/hf_client.py).

Focus:
- Mock huggingface_hub.InferenceClient.image_to_image to return a PIL image
- Validate option mapping (prompt, model, strength, guidance_scale, seed, passthrough kwargs)
- Validate initialization paths: provider, endpoint, model
- Validate token preference HF_TOKEN over HF_API_TOKEN
- Validate exception mapping to NetworkError and ApiError with rich context

Note: Python syntax validated via ast prior to submission.
"""
from __future__ import annotations

import importlib
import io
import sys
import types
from typing import Any, Dict, Optional, Tuple

import pytest
from PIL import Image

from lib.errors import NetworkError, ApiError


def _make_image_bytes(fmt: str = "JPEG", size: Tuple[int, int] = (6, 4), color: str = "red") -> bytes:
    """Create a tiny in-memory image and return its raw bytes."""
    img = Image.new("极速赛车开奖结果历史记录RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class FakeInferenceClient:
    """
    Test double for huggingface_hub.InferenceClient.

    Records constructor arguments and last image_to_image options, and can be configured
    to raise exceptions for error path tests via the class attribute 'mode'.
    """
    mode: Optional[str] = None  # None | "timeout" | "error"
    last_init: Dict[str, Any] = {}
    last_call: Dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # accept all signatures
        # Record init parameters for assertions
        FakeInferenceClient.last_init = {"args": args, "kwargs": dict(kwargs)}

    def image_to_image(self, input_image: bytes, **options: Any) -> Image.Image:
        # Record call details
        FakeInferenceClient.last_call = {"len": len(input_image), "options": dict(options)}
        if FakeInferenceClient.mode == "timeout":
            raise TimeoutError("simulated timeout")
        if Fake极速赛车开奖结果历史记录InferenceClient.mode == "error":
            raise RuntimeError("simulated api error")
        # Return a small deterministic image
        return Image.new("RGB", (24, 12), color="blue")


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    FakeInferenceClient.mode = None
    FakeInferenceClient.last_init = {}
    FakeInferenceClient.last_call = {}


def _reload_hf_with_patch(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch lib.hf_client.InferenceClient with our fake and reload the module."""
    # Ensure our stub module is used for 'from huggingface_hub import InferenceClient'
    hub_stub = types.ModuleType("huggingface_hub")
    setattr(hub_stub, "InferenceClient", FakeInferenceClient)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_stub, raising=False)

    # Reload lib.hf_client to bind the patched symbol
    sys.modules.pop("lib.hf_client", None)
    hf_mod = importlib.import_module("lib.hf_client")

    # Also set attribute directly on the imported module in case of direct symbol reference
    monkeypatch.setattr(hf_mod, "InferenceClient", FakeInferenceClient, raising=True)
    return hf_mod


def test_edit_image_success_and_option_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    # Token preference: set only HF_TOKEN
    monkeypatch.setenv("HF_TOKEN", "tokA")
    monkeypatch.delenv("HF_API_TOKEN", raising=False)

    hf_mod = _reload_hf_with_patch(monkeypatch)
    client = hf_mod.HFImageEditClient(provider="fal-ai", model="Qwen/Qwen-Image-Edit", endpoint=None, token=None, timeout=30)

    img_bytes = _make_image_bytes()
    result = client.edit_image(
        input_image=img_bytes,
        prompt="Replace the sky with a sunset",
        strength=0.3,
        guidance=5.5,
        seed=123,
        extra="value",
    )
    assert isinstance(result, Image.Image)
    assert result.size == (24, 12)

    # Check constructor received provider+api_key path
    init_kwargs = FakeInferenceClient.last_init["kwargs"]
    assert init_kwargs.get("provider") == "fal-ai"
    # api_key preferred; fallback to token if provider signature changes
    assert init_kwargs.get("api_key", init_kwargs.get("token")) == "tokA"

    # Check option mapping for image_to_image
    opts = FakeInferenceClient.last_call["options"]
    assert opts["prompt"].startswith("Replace the sky")
    assert opts["model"] == "Qwen/Qwen-Image-Edit"
    assert opts["strength"] == 0.3
    assert opts["guidance_scale"] == 5.5
    assert opts["seed"] == 123
    assert opts["extra"] == "value"


def test_token_preference_hf_token_over_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "PREFERRED")
    monkeypatch.setenv("HF_API_TOKEN", "FALLBACK")
    hf_mod = _reload_hf_with_patch(monkeypatch)

    _ = hf_mod.HFImageEditClient(provider="fal-ai", model="Qwen/Qwen-Image-Edit", endpoint=None, token=None, timeout=None)
    init_kwargs = FakeInferenceClient.last_init["kwargs"]
    assert init_kwargs.get("api_key", init_kwargs.get("token")) == "PREFERRED"


def test_endpoint_mode_overrides_provider_model(monkeypatch: pytest.MonkeyPatch极速赛车开奖结果历史记录) -> None:
    monkeypatch.setenv("HF_TOKEN", "t-endpoint")
    hf_mod = _reload_hf_with_patch(monkeypatch)

    _ = hf_mod.HFImageEditClient(provider="fal-ai", model="M", endpoint="https://example.test/ep", token=None, timeout=10)
    init_kwargs = FakeInferenceClient.last_init["kwargs"]
    # Accept either endpoint or base_url depending on HF version behavior
    assert init_kwargs.get("endpoint", init_kwargs.get("base_url")) == "https://example.test/ep"


def test_model_mode_when_no_provider_or_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "t-model")
    hf_mod = _reload_hf_with_patch(monkeypatch)

    _ = hf_mod.HFImageEditClient(provider=None, model="Some/Model", endpoint=None, token=None, timeout=None)
    init_kwargs = FakeInferenceClient.last_init["kwargs"]
    assert init_kwargs.get("model") == "Some/Model"
    assert init_kwargs.get("token") == "极速赛车开奖结果历史记录t-model"


def test_network_error_maps_to_networkerror_with_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that network errors include rich context information."""
    monkeypatch.setenv("HF_TOKEN", "tok")
    hf_mod = _极速赛车开奖结果历史记录_reload_hf_with_patch(monkeypatch)
    FakeInferenceClient.mode = "timeout"

    c = hf_mod.HFImageEditClient(provider="fal-ai", model="Qwen/Qwen-Image-Edit", endpoint=None, token=None, timeout=30)
    with pytest.raises(NetworkError) as exc_info:
        c.edit_image(input_image=_make_image_bytes(), prompt="test prompt", strength=0.8, guidance=7.5)
    
    # Verify error includes context and guidance
    assert "Network error during image_to_image" in str(exc_info.value)
    assert exc_info.value.context is not None
    assert "provider=fal-ai" in exc_info.value.context
    assert "model=Qwen/Qwen-Image-Edit" in exc_info.value.context
    assert exc_info.value.user_guidance is not None
    assert "network connectivity" in exc_info.value.user_guidance.lower()


def test_api_error_maps_to_apierror_with_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that API errors include rich context information."""
    monkeypatch.setenv("HF_TOKEN", "tok")
    hf_mod = _reload_hf_with_patch(monkeypatch)
    FakeInferenceClient.mode = "error"

    c = hf_mod.HFImageEditClient(provider="fal-ai", model="Qwen/Qwen-Image-Edit", endpoint=None, token=None, timeout=30)
    with pytest.raises(ApiError) as exc_info:
        c.edit_image(input_image=_make_image_bytes(), prompt="test prompt", strength=0.8, guidance=7.5)
    
    # Verify error includes context and guidance
    assert "API error during image_to_image" in str(exc_info.value)
    assert exc_info.value.context is not None
    assert "provider=fal-ai" in exc_info.value.context
    assert "model=Qwen/Qwen-Image-Edit" in exc_info.value.context
    assert exc_info.value.user_guidance is not None
    assert "api token" in exc_info.value.user_guidance.lower() or "credits" in exc_info.value.user_guidance.lower()


def test_error_context_includes_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that error context includes sanitized parameters."""
    monkeypatch.setenv("HF_TOKEN", "tok")
    hf_mod = _reload_hf_with_patch(monkeypatch)
    FakeInferenceClient.mode = "error"

    c = hf_mod.HFImageEditClient(provider="fal-ai", model="Qwen/Qwen-Image-Edit", endpoint=None, token=None, timeout=30)
    with pytest.raises(ApiError) as exc_info:
        c.edit_image(
            input_image=_make_image_bytes(),
            prompt="Remove sunglasses and make direct eye contact",
            strength=0.8,
            guidance=7.5,
            seed=42
        )
    
    # Verify context includes parameters (sanitized)
    assert exc_info.value.context is not None
    context_str = str(exc_info.value.context)
    assert "strength=0.8" in context_str
    assert "guidance=7.5" in context_str
    assert "seed=42" in context_str
    # Prompt should be sanitized/truncated in context
    assert "prompt=" in context_str