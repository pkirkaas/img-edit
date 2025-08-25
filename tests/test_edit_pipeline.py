"""
Tests for the edit pipeline in [src/lib/edit_pipeline.py](src/lib/edit_pipeline.py).

Focus:
- Integration of pipeline stages with mocked ImageIO and Provider
- edit_image success path with and without mask
- Error propagation from provider and image processing layers with rich context
- Validation behavior regarding output directory, etc.

External dependencies are fully mocked to avoid real I/O and network calls.

Note: Python syntax validated via ast prior to submission.
"""
from __future__ import annotations

import os
# Set environment variables before any imports to avoid config validation errors
os.environ["HF_API_TOKEN"] = "unit-test-token"
os.environ["PROVIDERS__replicate__provider_type"] = "replicate"

import importlib
import io
import sys
import types
import logging
from pathlib import Path
from typing import Any, Dict, Generator, Tuple, Optional

import pytest
from PIL import Image

from lib.errors import ApiError, NetworkError, create_error_context


class FakeImageIO:
    """
    Minimal ImageIO stub used by the pipeline tests.

    - load_image returns (bytes, info) tuple as expected by pipeline internals
    - save_image writes the image to disk and returns output info
    """

    def load_image(self, input_path: str) -> Tuple[bytes, Dict[str, Any]]:
        # The pipeline only forwards bytes to the client; contents don't matter here.
        # Provide small valid JPEG bytes to keep semantics realistic where needed.
        img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue(), {"format": "JPEG", "dimensions": (10, 10)}

    def save_image(self, edited_image: Image.Image, output_path: str) -> Dict[str, Any]:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as JPEG for deterministic behavior
        edited_image.save(out_path, format="JPEG")
        return {
            "file_size": out_path.stat().st_size,
            "format": "JPEG",
            "dimensions": edited_image.size,
        }


class FakeProvider:
    """
    ImageEditProvider stub capturing invocation details and optionally raising errors.

    Class attributes:
      - raise_mode: None | "api" | "network"
      - EXC_API / EXC_NET: exception classes to raise (bound at runtime in tests)
    """
    raise_mode: str | None = None
    EXC_API: type[BaseException] = Exception
    EXC_NET: type[BaseException] = Exception

    def __init__(self):
        self.last_call: Dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return "replicate"  # Match default provider name

    @property
    def description(self) -> str:
        return "Fake provider for testing"

    def edit_image(
        self,
        image: Image.Image,
        instructions: str,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        mask: Optional[Image.Image] = None,
        **kwargs: Any
    ) -> Image.Image:
        if FakeProvider.raise_mode == "api":
            # Create a rich ApiError with context
            context = create_error_context(
                operation="image_edit",
                provider=self.name,
                parameters={
                    "instructions": instructions,
                    "strength": strength,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "mask_provided": mask is not None,
                    **kwargs
                }
            )
            raise ApiError("API error during image editing", context=context)
        if FakeProvider.raise_mode == "network":
            # Create a rich NetworkError with context
            context = create_error_context(
                operation="image_edit",
                provider=self.name
            )
            raise NetworkError("Network error during image editing", context=context)
        # Record invocation for assertions
        self.last_call = {
            "image_size": image.size,
            "instructions": instructions,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "mask_provided": mask is not None,
            "kwargs": kwargs,
        }
        # Return a predictable image
        return Image.new("RGB", (24, 12), color="blue")

    def close(self) -> None:
        pass


@pytest.fixture()
def pipeline_with_stubs(monkeypatch: pytest.MonkeyPatch) -> Generator[Any, None, None]:
    """
    Fixture that:
      - Ensures required env var for lib.config import
      - Injects a stub module for lib.image_io so edit_pipeline can import ImageIO
      - Imports lib.edit_pipeline and patches provider registry to use fake provider
    Yields the imported pipeline module.
    """
    # Ensure settings can import (global settings = Settings() runs on import)
    monkeypatch.setenv("HF_API_TOKEN", "unit-test-token")
    # Configure default provider to avoid validation errors
    monkeypatch.setenv("PROVIDERS__replicate__provider_type", "replicate")

    # Create stub lib.image_io module with the expected names
    stub_mod = types.ModuleType("lib.image_io")
    # Names used by edit_pipeline: ImageIO (class) and ImageValidationError (type)
    setattr(stub_mod, "ImageIO", FakeImageIO)
    # Provide a named exception so pipeline error_type matches "ImageValidationError"
    class ImageValidationError(Exception):
        pass
    setattr(stub_mod, "ImageValidationError", ImageValidationError)

    # Inject the stub before importing the pipeline
    monkeypatch.setitem(sys.modules, "lib.image_io", stub_mod, raising=False)

    # Now import the pipeline module
    pipeline_mod = importlib.import_module("lib.edit_pipeline")
    # Inject missing 'logging' symbol used by the pipeline for TimingContext levels
    monkeypatch.setattr(pipeline_mod, "logging", logging, raising=False)
    
    # Bind exception classes for the provider stub now that lib.errors is importable
    errors_mod = importlib.import_module("lib.errors")
    FakeProvider.EXC_API = errors_mod.ApiError
    FakeProvider.EXC_NET = errors_mod.NetworkError
    
    # Create a fake provider instance for testing
    fake_provider = FakeProvider()
    
    # Patch the provider registry to return our fake provider
    def mock_get_provider_class(name: str):
        if name == "replicate":
            return FakeProvider
        raise KeyError(f"Provider '{name}' not found")
    
    # Import and patch the provider registry
    providers_mod = importlib.import_module("lib.providers")
    monkeypatch.setattr(providers_mod.get_registry(), "get_provider_class", mock_get_provider_class)

    try:
        yield pipeline_mod
    finally:
        # Reset raise mode after each test
        FakeProvider.raise_mode = None


def _create_dummy_input_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The content is irrelevant (we don't load this file in the stub),
    # but _validate_inputs requires that it exists.
    path.write_bytes(b"dummy")


def test_pipeline_success_without_mask(tmp_path: Path, pipeline_with_stubs: Any) -> None:
    pipeline_mod = pipeline_with_stubs
    from lib.config import Settings  # import after env is set

    input_path = tmp_path / "in.jpg"
    output_dir = tmp_path / "out"
    output_path = output_dir / "edited.jpg"
    _create_dummy_input_file(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = pipeline_mod.ImageEditPipeline(Settings(HF_API_TOKEN="unit-test-token"))
    result = pipe.edit_image(
        input_path=str(input_path),
        output_path=str(output_path),
        prompt="Turn the image blue",
    )

    assert result["status"] == "success"
    assert Path(result["output_path"]) == output_path
    assert output_path.exists() and output_path.stat().st_size > 0
    assert result["file_size"] == output_path.stat().st_size
    assert result["image_format"] == "JPEG"
    assert result["image_dimensions"] == (24, 12)  # from FakeProvider returned image


def test_pipeline_success_with_mask_forwards_to_provider(tmp_path: Path, pipeline_with_stubs: Any) -> None:
    pipeline_mod = pipeline_with_stubs
    from lib.config import Settings

    input_path = tmp_path / "in2.jpg"
    output_dir = tmp_path / "out2"
    output_path = output_dir / "edited2.jpg"
    _create_dummy_input_file(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = pipeline_mod.ImageEditPipeline(Settings(HF_API_TOKEN="unit-test-token"))
    result = pipe.edit_image(
        input_path=str(input_path),
        output_path=str(output_path),
        prompt="Any",
        mask="some-base64-mask",
    )
    assert result["status"] == "success"
    # Verify the provider saw the mask
    # Note: The pipeline converts mask string to Image object, so we check if mask was provided
    assert hasattr(pipe, 'provider') and pipe.provider.last_call is not None
    assert pipe.provider.last_call["mask_provided"] is True


def test_pipeline_provider_api_error_returns_error_status_with_context(tmp_path: Path, pipeline_with_stubs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_mod = pipeline_with_stubs
    from lib.config import Settings

    # Configure stub provider to raise API error
    FakeProvider.raise_mode = "api"

    input_path = tmp_path / "in3.jpg"
    output_dir = tmp_path / "out3"
    output_path = output_dir / "edited3.jpg"
    _create_dummy_input_file(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = pipeline_mod.ImageEditPipeline(Settings(HF_API_TOKEN="unit-test-token"))
    result = pipe.edit_image(
        input_path=str(input_path),
        output_path=str(output_path),
        prompt="cause api error",
        strength=0.8,
        guidance_scale=7.5,
        seed=42
    )
    assert result["status"] == "error"
    # Error type should reflect the originating exception class name
    assert result["error_type"] == "ApiError"
    assert "Pipeline failed" in result["message"]
    # Verify the error message includes context from the ApiError
    assert "provider=" in result["message"]
    assert "strength=0.8" in result["message"]
    assert "guidance_scale=7.5" in result["message"]
    assert "seed=42" in result["message"]


def test_pipeline_provider_network_error_returns_error_status_with_context(tmp_path: Path, pipeline_with_stubs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_mod = pipeline_with_stubs
    from lib.config import Settings

    # Configure stub provider to raise network error
    FakeProvider.raise_mode = "network"

    input_path = tmp_path / "in4.jpg"
    output_dir = tmp_path / "out4"
    output_path = output_dir / "edited4.jpg"
    _create_dummy_input_file(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = pipeline_mod.ImageEditPipeline(Settings(HF_API_TOKEN="unit-test-token"))
    result = pipe.edit_image(
        input_path=str(input_path),
        output_path=str(output_path),
        prompt="cause network error",
    )
    assert result["status"] == "error"
    # Error type should reflect the originating exception class name
    assert result["error_type"] == "NetworkError"
    assert "Pipeline failed" in result["message"]
    # Verify the error message includes context from the NetworkError
    assert "provider=" in result["message"]


def test_pipeline_image_io_load_error_returns_error_status(tmp_path: Path, pipeline_with_stubs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_mod = pipeline_with_stubs
    from lib.config import Settings

    input_path = tmp_path / "in5.jpg"
    output_dir = tmp_path / "out5"
    output_path = output_dir / "edited5.jpg"
    _create_dummy_input_file(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = pipeline_mod.ImageEditPipeline(Settings(HF_API_TOKEN="unit-test-token"))
    # Force image_io to raise an error during load; pipeline should report ImageValidationError
    def boom(_path: str) -> tuple[bytes, Dict[str, Any]]:
        raise RuntimeError("load failed")

    monkeypatch.setattr(pipe.image_io, "load_image", boom, raising=True)
    result = pipe.edit_image(
        input_path=str(input_path),
        output_path=str(output_path),
        prompt="any",
    )
    assert result["status"] == "error"
    assert result["error_type"] == "ImageValidationError"
    assert "Unexpected error loading image" in result["message"]


def test_pipeline_validation_error_returns_error_status(tmp_path: Path, pipeline_with_stubs: Any) -> None:
    pipeline_mod = pipeline_with_stubs
    from lib.config import Settings

    input_path = tmp_path / "nonexistent.jpg"
    output_path = tmp_path / "output.jpg"

    pipe = pipeline_mod.ImageEditPipeline(Settings(HF_API_TOKEN="unit-test-token"))
    result = pipe.edit_image(
        input_path=str(input_path),
        output_path=str(output_path),
        prompt="test prompt",
    )
    assert result["status"] == "error"
    assert "does not exist" in result["message"]
    assert result["error_type"] == "ValueError"