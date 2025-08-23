"""
Image input/output operations for the image editing application.

This module provides functions for loading, saving, and validating images using the Pillow library.
It handles common image formats and provides robust error handling for file operations.

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
from pathlib import Path
from typing import Optional, Union

from PIL import Image, ImageFile, UnidentifiedImageError

# Enable loading of truncated images (useful for some corrupt but recoverable images)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Supported image formats for input and output
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "BMP", "GIF", "TIFF", "WEBP"}


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
        raise FileNotFoundError(f"Image file not found: {path}")
    
    # Check if file is readable
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Permission denied: cannot read file {path}")
    
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
        raise UnidentifiedImageError(f"Could not identify image file {path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading image {path}: {e}")


def save_image(image: Image.Image, path: Union[str, Path], **kwargs) -> None:
    """
    Save an image to the specified file path with format detection.
    
    This function saves an image to disk, automatically detecting the format
    from the file extension and handling appropriate saving parameters.
    
    Args:
        image: PIL Image object to save
        path: Destination file path (string or Path object)
        **kwargs: Additional arguments passed to PIL Image.save()
        
    Raises:
        ValueError: If the file format is not supported
        PermissionError: If the file cannot be written due to permissions
        RuntimeError: If the image cannot be saved
        
    Example:
        >>> save_image(image, "output.png", quality=95)
        >>> save_image(image, "result.jpg", optimize=True)
    """
    path = Path(path) if isinstance(path, str) else path
    
    # Check if directory exists and is writable
    parent_dir = path.parent
    if parent_dir and not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
    
    if parent_dir and not os.access(parent_dir, os.W_OK):
        raise PermissionError(f"Permission denied: cannot write to directory {parent_dir}")
    
    # Get file extension and validate format
    ext = path.suffix.lower().lstrip(".")
    if ext.upper() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {ext}. Supported formats: {SUPPORTED_FORMATS}")
    
    try:
        # Set appropriate save parameters based on format
        save_args = {}
        if ext in ("jpg", "jpeg"):
            save_args.setdefault("quality", 95)
            save_args.setdefault("optimize", True)
        elif ext == "png":
            save_args.setdefault("optimize", True)
        
        # Merge with any provided kwargs
        save_args.update(kwargs)
        
        # Save the image
        image.save(path, format=ext.upper(), **save_args)
        
    except Exception as e:
        raise RuntimeError(f"Error saving image to {path}: {e}")


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
        raise FileNotFoundError(f"Mask file not found: {path}")
    
    try:
        # Load the mask image
        mask = load_image(path)
        
        # Ensure mask is in appropriate mode (RGBA for transparency)
        if mask.mode != "RGBA":
            mask = mask.convert("RGBA")
            
        return mask
        
    except Exception as e:
        raise RuntimeError(f"Error loading mask {path}: {e}")


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
        raise ValueError(
            f"Unsupported image format: {format}. "
            f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
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