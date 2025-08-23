"""
Typer-based CLI package for the image editing application.

This package contains the command-line interface implementation built with Typer.
The main entry point is implemented in [src/cli/main.py](src/cli/main.py).

Integration points:
- [src/lib/config.py](src/lib/config.py): environment-backed settings
- [src/lib/image_io.py](src/lib/image_io.py): image read/write helpers
- [src/lib/hf_client.py](src/lib/hf_client.py): Hugging Face API client
- [src/lib/edit_pipeline.py](src/lib/edit_pipeline.py): image edit pipeline facade
- [src/lib/logging_utils.py](src/lib/logging_utils.py): structured logging

The CLI is executed via the PDM script "img-edit" configured in [pyproject.toml](pyproject.toml).
"""

__all__ = ["__doc__"]