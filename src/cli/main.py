"""
Typer-based CLI entry point for the Image Edit application.

This CLI exposes an "edit" command to send an image and prompt to the Hugging Face
Serverless Inference API for Qwen/Qwen-Image-Edit. It integrates with the core library
pipeline and configuration modules, and is intended to be executed via the PDM script
"img-edit" defined in [pyproject.toml](pyproject.toml).

Execution:
- Use PDM (see [docs/setup.md](docs/setup.md))
- Example help: `pdm run img-edit --help`
- Example command (see [docs/usage-cli.md](docs/usage-cli.md))

Integration points (core library):
- [src/lib/config.py](src/lib/config.py)
- [src/lib/edit_pipeline.py](src/lib/edit_pipeline.py)
- [src/lib/hf_client.py](src/lib/hf_client.py)
- [src/lib/image_io.py](src/lib/image_io.py)
- [src/lib/logging_utils.py](src/lib/logging_utils.py)

Notes:
- Imports of lib.* are deferred into command handlers to avoid raising configuration
  errors when running `--help` on systems without required environment variables set.
- Errors are reported with clear messages, while structured logs go to the logger.

This file's Python syntax has been validated with ast prior to submission.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import typer


# Typer application instance (no shell completion for compactness here)
app = typer.Typer(
    add_completion=False,
    help="AI Image Edit CLI using Hugging Face Serverless Inference (Qwen/Qwen-Image-Edit).",
)


def _encode_file_b64(path: Path) -> str:
    """
    Read a file from disk and return its Base64-encoded string.

    This helper is used to pass optional mask images to the pipeline/client as an
    additional input field in the API payload. The core HF client will transparently
    merge any extra inputs provided via kwargs.

    Args:
        path: Path to the file that should be Base64 encoded

    Returns:
        Base64-encoded string of the file content (UTF-8)

    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be read due to permissions
        OSError: For other IO related errors

    Example:
        >>> encoded = _encode_file_b64(Path("mask.png"))
        >>> assert isinstance(encoded, str)
    """
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


@app.command("edit")
def edit(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to input image file",
        metavar="FILE",
    ),
    prompt: str = typer.Option(
        ...,
        "--prompt",
        "-p",
        help="Text prompt describing the requested edit",
        metavar="TEXT",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        "-o",
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
        help="Path to save the edited image",
        metavar="FILE",
    ),
    mask: Optional[Path] = typer.Option(
        None,
        "--mask",
        "-m",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Optional path to a mask image file. Mask will be sent as base64 in the request.",
        metavar="FILE",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Optional seed for deterministic results",
        metavar="INT",
    ),
    strength: Optional[float] = typer.Option(
        None,
        "--strength",
        help="Strength parameter for editing (0.0 - 1.0). Defaults to config if omitted.",
        min=0.0,
        max=1.0,
        metavar="FLOAT",
    ),
    guidance: Optional[float] = typer.Option(
        None,
        "--guidance",
        help="Guidance scale (creativity vs adherence). Defaults to config if omitted.",
        min=0.0,
        metavar="FLOAT",
    ),
    timeout: Optional[int] = typer.Option(
        None,
        "--timeout",
        help="HTTP timeout in seconds (overrides config for this run)",
        min=1,
        metavar="SECONDS",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help="Hugging Face inference endpoint URL (overrides config for this run)",
        metavar="URL",
    ),
) -> None:
    """
    Edit an image using the Qwen/Qwen-Image-Edit model via the Hugging Face Serverless API.

    Parameters:
        input_path: Path to the input image to be edited
        prompt: Text prompt describing the edit to apply
        output_path: Destination path for the edited image
        mask: Optional path to a mask image; if provided, it is Base64-encoded and sent as 'mask'
        seed: Optional seed for deterministic generation
        strength: Optional strength for editing; falls back to configuration default if omitted
        guidance: Optional guidance scale; falls back to configuration default if omitted
        timeout: Optional request timeout override; if provided, overrides configuration for this invocation
        endpoint: Optional Hugging Face endpoint override; if provided, overrides configuration for this invocation

    Behavior:
        - Ensures the output directory exists before invoking the pipeline
        - Constructs a Settings object honoring optional endpoint/timeout overrides
        - Invokes the pipeline's edit operation with provided parameters
        - Emits user-friendly messages and returns appropriate process exit codes

    Raises:
        typer.Exit: With non-zero status if an error occurs
    """
    # Defer library imports to avoid config evaluation during --help
    try:
        from lib.config import Settings  # type: ignore
        from lib.edit_pipeline import ImageEditPipeline, PipelineError  # type: ignore
        from lib.hf_client import HFAPIError, HFNetworkError  # type: ignore
        from lib.logging_utils import get_logger  # type: ignore
    except Exception as e:
        typer.secho(f"Failed to import application modules: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    logger = get_logger(__name__)

    # Ensure output directory exists to satisfy pipeline validation
    out_dir = output_path.parent
    try:
        if out_dir and not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        typer.secho(f"Cannot create output directory '{out_dir}': {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Optional per-run config overrides
    settings_overrides = {}
    if timeout is not None:
        settings_overrides["IMG_EDIT_TIMEOUT_SECONDS"] = timeout
    if endpoint is not None:
        settings_overrides["HF_INFERENCE_ENDPOINT"] = endpoint

    # Build settings (validates env and provided overrides)
    try:
        settings = Settings(**settings_overrides)
    except Exception as e:
        typer.secho(
            f"Configuration error: {e}. Ensure .env is configured (HF_API_TOKEN required).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    # Prepare extra payload args
    extra_inputs = {}
    try:
        if mask is not None:
            # Read and encode mask as base64 string so the HF client can include it in JSON payload
            extra_inputs["mask"] = _encode_file_b64(mask)
    except Exception as e:
        typer.secho(f"Failed to read/encode mask '{mask}': {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Execute pipeline
    try:
        logger.info("Starting image edit operation")
        pipeline = ImageEditPipeline(settings)
        result = pipeline.edit_image(
            input_path=str(input_path),
            output_path=str(output_path),
            prompt=prompt,
            guidance_scale=guidance,
            strength=strength,
            seed=seed,
            **extra_inputs,  # passed to HF client as additional inputs
        )

        status = str(result.get("status", "error"))
        message = str(result.get("message", ""))
        if status == "success":
            typer.secho(f"{message}", fg=typer.colors.GREEN)
            typer.echo(f"Output: {result.get('output_path')}")
            typer.echo(f"Image: {result.get('image_format')} {result.get('image_dimensions')}")
            typer.echo(f"Size: {result.get('file_size')} bytes")
            typer.echo(f"Time: {result.get('execution_time'):.2f}s")
            raise typer.Exit(code=0)
        else:
            typer.secho(f"Edit failed: {message}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    except (HFAPIError, HFNetworkError, PipelineError, FileNotFoundError, PermissionError, ValueError) as e:
        typer.secho(f"{type(e).__name__}: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        # Catch-all for unexpected errors
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


def main() -> None:
    """
    CLI program entry point for execution via `python -m cli.main`.

    This function simply delegates to the Typer application instance.

    Example:
        $ pdm run img-edit --help
    """
    app()


if __name__ == "__main__":
    main()