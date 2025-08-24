"""
Image input/output operations for the image editing application.

This module provides functions for loading, saving, and validating images using the Pillow library.
It handles common image formats and provides robust error handling for file operations with rich
context and user guidance for better diagnostics.

Supported image formats:
    - JPEG/JPG
    - PNG
    - BMP
    - GIF
    - TIFF
    - WebP

Example usage:
    >>> from lib.image_io import load_image, save_image
    >>> image = load_image("input.jpg")
    >>> save_image(image, "output.png")

Functions:
    load_image: Load an image from file path with validation
    save_image: Save an image to file path with format detection
    load_mask: Load an optional mask image (supports transparency)
    validate_image_format: Validate that an image is in a supported format
"""

import os
import io
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, Any

from PIL import Image, ImageFile, UnidentifiedImageError

from lib.errors import ValidationError, create_error_context

# Enable loading of truncated images (useful for some corrupt but recoverable images)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Supported image formats for input and output
SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "GIF", "TIFF", "WEBP"}

# Mapping from filename extension to Pillow format names
EXT_TO_PILLOW_FORMAT = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
    "bmp": "BMP",
    "gif": "GIF",
    "tif": "TIFF",
    "tiff": "TIFF",
}

# Public API of this module
__all__ = [
    "SUPPORTED_FORMATS",
    "load_image",
    "save_image",
    "load_mask",
    "validate_image_format",
    "ImageIO",
    "ImageValidationError",
]


class ImageValidationError(ValidationError):
    """
    Exception type for image validation or loading issues encountered by ImageIO.

    Raised when an image cannot be found, read, identified, or is in an unsupported format
    for the purposes of the editing pipeline. Includes rich context and user guidance.

    Attributes:
        context: Error context with operation details
        user_guidance: Helpful guidance for resolving the issue

    Example:
        >>> raise ImageValidationError("Unsupported image format: ICO", context=ctx, user_guidance="Use JPEG or PNG")
    """
    pass


class ImageIO:
    """
    Object-oriented facade over module-level image I/O helpers, tailored for the pipeline.

    Methods:
      - load_image(path): returns (bytes, info) where:
            info = { "format": "PNG"|"JPEG", "dimensions": (width, height) }
        Uses PNG when the image has an alpha channel, otherwise JPEG.
      - save_image(image, path): persists the image and returns:
            { "file_size": int, "format": str, "dimensions": (w, h) }

    Notes:
      - Internally delegates to the module-level functions for validation and saving.
      - Converts images to bytes for transport to the HF client.
    """

    def load_image(self, path: Union[str, Path]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Load and validate an image, returning encoded bytes and simple metadata.

        Args:
            path: Path to the image file (str or Path)

        Returns:
            Tuple[bytes, Dict[str, Any]]:
                - bytes: Serialized image bytes (PNG if alpha, else JPEG)
                - info:  {"format": "PNG"|"JPEG", "dimensions": (width, height)}

        Raises:
            ImageValidationError: On not found, unreadable, unidentifiable, or unsupported format
        """
        try:
            # Reuse the robust module-level loader which validates existence/format
            img = load_image(path)
            # Preserve transparency using PNG; otherwise use JPEG for compactness
            encode_format = "PNG" if img.mode == "RGBA" else "JPEG"
            buf = io.BytesIO()
            img.save(buf, format=encode_format)
            return buf.getvalue(), {"format": encode_format, "dimensions": img.size}
        except (FileNotFoundError, PermissionError, UnidentifiedImageError, ValueError) as e:
            error_context = create_error_context(
                operation="image_loading",
                file_path=str(path),
                root_cause=e
            )
            raise ImageValidationError(
                f"Failed to load image: {e}",
                context=error_context,
                user_guidance="Check that the file exists, is readable, and is in a supported format (JPEG, PNG, BMP, GIF, TIFF, WebP)."
            ) from e
        except Exception as e:
            error_context = create_error_context(
                operation="image_loading",
                file_path=str(path),
                root_cause=e
            )
            raise ImageValidationError(
                f"Unexpected error loading image: {e}",
                context=error_context,
                user_guidance="This may indicate a corrupted file or system issue. Try with a different image file."
            ) from e

    def save_image(self, edited_image: Image.Image, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Save a PIL image to disk and return output metadata.

        Args:
            edited_image: PIL Image to save
            path: Destination file path (str or Path)

        Returns:
            Dict[str, Any]:
                {
                    "file_size": output size in bytes,
                    "format": stored image format (derived from file extension),
                    "dimensions": (width, height)
                }

        Raises:
            ValueError, PermissionError, RuntimeError: propagated from save operation
        """
        # Delegate persistence and return info from the module-level function
        info = save_image(edited_image, path)
        return info
def load_image(path: Union[str, Path]) -> Image.Image:
    """
    Load an image from the specified file path with validation.
    
    This function loads an image from disk, validates the file exists and is readable,
    and ensures the image format is supported.
    
    Args:
        path: Path to the image file (string or Path object)
        
    Returns:
        PIL.Image.Image: The loaded image object
        
    Raises:
        FileNotFoundError: If the image file does not exist
        PermissionError: If the image file cannot be read due to permissions
        ValueError: If the image format is not supported
        UnidentifiedImageError: If the image cannot be identified or is corrupt
        
    Example:
        >>> image = load_image("photo.jpg")
        >>> print(image.size)
        (800, 600)
    """
    path = Path(path) if isinstance(path, str) else path
    
    # Check if file exists
    if not path.exists():
        error_context = create_error_context(
            operation="image_loading",
            file_path=str(path),
            root_cause=FileNotFoundError()
        )
        raise ImageValidationError(
            f"Image file not found: {path}",
            context=error_context,
            user_guidance="Check the file path and ensure the image exists at the specified location."
        )
    
    # Check if file is readable
    if not os.access(path, os.R_OK):
        error_context = create_error_context(
            operation="image_loading",
            file_path=str(path),
            root_cause=PermissionError()
        )
        raise ImageValidationError(
            f"Permission denied: cannot read file {path}",
            context=error_context,
            user_guidance="Check file permissions and ensure the application has read access to the file."
        )
    
    try:
        # Open the image
        image = Image.open(path)
        
        # Validate the image format
        validate_image_format(image)
        
        # Convert to RGB if necessary (some formats like PNG might have alpha)
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
            
        return image
        
    except UnidentifiedImageError as e:
        error_context = create_error_context(
            operation="image_loading",
            file_path=str(path),
            root_cause=e
        )
        raise ImageValidationError(
            f"Could not identify image file {path}: {e}",
            context=error_context,
            user_guidance="The file may be corrupted or not a valid image. Try with a different image file."
        ) from e
    except Exception as e:
        error_context = create_error_context(
            operation="image_loading",
            file_path=str(path),
            root_cause=e
        )
        raise ImageValidationError(
            f"Error loading image {path}: {e}",
            context=error_context,
            user_guidance="This may indicate a system or file corruption issue. Verify the image file integrity."
        ) from e


def save_image(image: Image.Image, path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """
    Save an image to the specified file path with format detection and normalization.

    This function saves an image to disk, normalizing common filename extensions to the
    Pillow format names, ensuring the output directory exists, and applying minimal,
    sensible defaults per format. Unknown extensions default to PNG.

    Args:
        image: PIL Image object to save
        path: Destination file path (string or Path object)
        **kwargs: Additional arguments forwarded to PIL Image.save()

    Returns:
        Dict[str, Any]: {
            "file_size": output size in bytes,
            "format": stored image format (normalized Pillow format name),
            "dimensions": (width, height)
        }

    Raises:
        PermissionError: If the destination directory is not writable
        RuntimeError: If the image cannot be saved for any reason (includes path, format, and original error)
    """
    path = Path(path) if isinstance(path, str) else path

    # Ensure destination directory exists and is writable
    parent_dir = path.parent
    if parent_dir and not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
    if parent_dir and not os.access(parent_dir, os.W_OK):
        raise PermissionError(f"Permission denied: cannot write to directory {parent_dir}")

    # Determine format from extension, default to PNG when unknown
    ext = path.suffix.lower().lstrip(".")
    fmt = EXT_TO_PILLOW_FORMAT.get(ext, "PNG")

    # Apply minimal defaults per format
    save_args: Dict[str, Any] = {}
    if fmt == "JPEG":
        save_args.setdefault("quality", 95)
        save_args.setdefault("optimize", True)
    elif fmt == "PNG":
        save_args.setdefault("optimize", True)

    # Merge any provided kwargs (caller wins)
    save_args.update(kwargs)

    # Ensure JPEG-compatible mode (JPEG does not support alpha or palette)
    img_to_save = image
    if fmt == "JPEG" and image.mode not in ("RGB",):
        img_to_save = image.convert("RGB")

    try:
        img_to_save.save(path, format=fmt, **save_args)
    except Exception as e:
        error_context = create_error_context(
            operation="image_saving",
            file_path=str(path),
            format=fmt,
            root_cause=e
        )
        raise ValidationError(
            f"Error saving image to {path}: {fmt}: {e}",
            context=error_context,
            user_guidance="Check that the destination directory exists and is writable. Verify available disk space."
        ) from e

    # Gather output metadata
    file_size = path.stat().st_size
    dimensions = img_to_save.size
    return {"file_size": file_size, "format": fmt, "dimensions": dimensions}


def load_mask(path: Union[str, Path]) -> Optional[Image.Image]:
    """
    Load an optional mask image for image editing operations.
    
    This function loads a mask image that can be used to specify which parts
    of an image should be edited. Supports transparency in mask images.
    
    Args:
        path: Path to the mask image file (string or Path object)
        
    Returns:
        Optional[PIL.Image.Image]: The loaded mask image, or None if path is None/empty
        
    Raises:
        FileNotFoundError: If the mask file does not exist
        PermissionError: If the mask file cannot be read
        ValueError: If the mask format is not supported
        UnidentifiedImageError: If the mask cannot be identified
        
    Example:
        >>> mask = load_mask("mask.png")
        >>> if mask:
        ...     print(f"Mask loaded: {mask.size}")
    """
    if not path:
        return None
    
    path = Path(path) if isinstance(path, str) else path
    
    # Check if file exists
    if not path.exists():
        error_context = create_error_context(
            operation="mask_loading",
            file_path=str(path),
            root_cause=FileNotFoundError()
        )
        raise ImageValidationError(
            f"Mask file not found: {path}",
            context=error_context,
            user_guidance="Check the mask file path and ensure it exists at the specified location."
        )
    
    try:
        # Load the mask image
        mask = load_image(path)
        
        # Ensure mask is in appropriate mode (RGBA for transparency)
        if mask.mode != "RGBA":
            mask = mask.convert("RGBA")
            
        return mask
        
    except Exception as e:
        error_context = create_error_context(
            operation="mask_loading",
            file_path=str(path),
            root_cause=e
        )
        raise ImageValidationError(
            f"Error loading mask {path}: {e}",
            context=error_context,
            user_guidance="Check that the mask file is a valid image in a supported format and is readable."
        ) from e


def validate_image_format(image: Image.Image) -> None:
    """
    Validate that an image is in a supported format.
    
    Args:
        image: PIL Image object to validate
        
    Raises:
        ValueError: If the image format is not supported
        
    Example:
        >>> validate_image_format(image)  # Raises ValueError if unsupported
    """
    format = image.format
    if format and format.upper() not in SUPPORTED_FORMATS:
        error_context = create_error_context(
            operation="image_validation",
            detected_format=format,
            supported_formats=list(SUPPORTED_FORMATS),
            root_cause=ValueError()
        )
        raise ImageValidationError(
            f"Unsupported image format: {format}",
            context=error_context,
            user_guidance=f"Supported formats: {sorted(SUPPORTED_FORMATS)}. Convert the image to a supported format before processing."
        )


if __name__ == "__main__":
    # Test the image IO functionality
    try:
        # Create a test image
        test_image = Image.new("RGB", (100, 100), color="red")
        
        # Test saving and loading
        save_image(test_image, "test_output.jpg")
        loaded = load_image("test_output.jpg")
        print(f"Image saved and loaded successfully: {loaded.size}")
        
        # Test mask loading (nonexistent should return None)
        mask = load_mask(None)
        print(f"None mask handled correctly: {mask is None}")
        
        # Clean up
        Path("test_output.jpg").unlink(missing_ok=True)
        
    except Exception as e:
        print(f"Test failed: {e}")