"""
Tests for image input/output operations in [src/lib/image_io.py](src/lib/image_io.py).

Covers:
- Image loading with valid and invalid paths
- Image saving in different formats
- Optional mask loading
- Image format validation (including unsupported formats)

Note: Python syntax validated via ast prior to submission.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from lib import image_io


def test_load_image_valid_path(tmp_path: Path) -> None:
    # Arrange
    img_path = tmp_path / "input.jpg"
    Image.new("RGB", (16, 9), color="red").save(img_path, format="JPEG")
    # Act
    img = image_io.load_image(img_path)
    # Assert
    assert img.size == (16, 9)
    assert img.mode in {"RGB", "RGBA"}


def test_load_image_invalid_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError):
        image_io.load_image(missing)


@pytest.mark.parametrize("ext,fmt", [("jpg", "JPEG"), ("png", "PNG")])
def test_save_image_different_formats(tmp_path: Path, ext: str, fmt: str) -> None:
    # Arrange
    out_path = tmp_path / f"saved.{ext}"
    img = Image.new("RGB", (10, 10), color="blue")
    # Act
    image_io.save_image(img, out_path)
    # Assert
    assert out_path.exists() and out_path.stat().st_size > 0
    # Verify reload succeeds and format matches the extension (best-effort; Pillow sets .format on opened file)
    with Image.open(out_path) as reloaded:
        assert reloaded.format == fmt


def test_load_mask_none_returns_none() -> None:
    assert image_io.load_mask(None) is None  # type: ignore[arg-type]


def test_load_mask_png_returns_rgba(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    # Create a grayscale mask; loader should convert to RGBA
    Image.new("L", (8, 8), color=255).save(mask_path, format="PNG")
    mask = image_io.load_mask(mask_path)
    assert mask is not None
    assert mask.mode == "RGBA"
    assert mask.size == (8, 8)


def test_validate_image_format_rejects_ico(tmp_path: Path) -> None:
    ico_path = tmp_path / "icon.ico"
    Image.new("RGB", (16, 16), color="green").save(ico_path, format="ICO")
    with Image.open(ico_path) as ico_img:
        assert ico_img.format == "ICO"
        with pytest.raises(ValueError):
            image_io.validate_image_format(ico_img)  # Unsupported by SUPPORTED_FORMATS


def test_load_image_rejects_unsupported_format(tmp_path: Path) -> None:
    # load_image internally calls validate_image_format; on unsupported formats it wraps as RuntimeError
    ico_path = tmp_path / "icon2.ico"
    Image.new("RGB", (16, 16), color="purple").save(ico_path, format="ICO")
    with pytest.raises(RuntimeError):
        image_io.load_image(ico_path)


def test_save_image_jpg_extension_maps_to_jpeg(tmp_path: Path) -> None:
    # Arrange
    out_path = tmp_path / "mapped.jpg"
    img = Image.new("RGB", (12, 8), color="orange")
    # Act
    info = image_io.save_image(img, out_path)
    # Assert
    assert out_path.exists() and out_path.stat().st_size > 0
    with Image.open(out_path) as reloaded:
        assert reloaded.format == "JPEG"
    assert info["format"] == "JPEG"
    assert info["dimensions"] == (12, 8)


def test_save_rgba_image_to_jpg_converts_and_succeeds(tmp_path: Path) -> None:
    # Arrange
    out_path = tmp_path / "alpha.jpg"
    rgba = Image.new("RGBA", (9, 7), color=(10, 20, 30, 128))
    # Act
    info = image_io.save_image(rgba, out_path)
    # Assert
    assert out_path.exists() and out_path.stat().st_size > 0
    with Image.open(out_path) as reloaded:
        assert reloaded.format == "JPEG"
        assert reloaded.mode == "RGB"
    assert info["format"] == "JPEG"