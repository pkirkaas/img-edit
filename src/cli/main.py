"""
Typer-based CLI entry point for the Image Edit application.

Syntax validated with ast.parse.

This CLI exposes a single-command interface that edits an image using AI.
It integrates with the core library pipeline and configuration modules, and
is intended to be executed via the PDM script "img-edit" defined in [pyproject.toml](pyproject.toml).

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
- [src/lib/errors.py](src/lib/errors.py)

Notes:
- Imports of lib.* are deferred into handlers to avoid raising configuration
  errors when running `--help` on systems without required environment variables set.
- Errors are reported with clear messages, while structured logs go to the logger.
- Supports --verbose and --debug flags for detailed error reporting.
- Saves error artifacts to tmp/last_error.json for debugging.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import typer


# Typer application instance (no shell completion for compactness here)
app = typer.Typer(
    add_completion=False,
    help=(
        "AI Image Edit CLI. Single-command interface: run with options to edit images. "
        "Use '--providers' to list available providers. "
        "Default provider: replicate; Hugging Face is fully supported. "
        "Run 'pdm run img-edit --help' for detailed options and examples."
    ),
)


@app.callback(invoke_without_command=True)
def _root_callback(
    # Edit options (all optional here so --providers can be used alone; validated below)
    input_path: Optional[Path] = typer.Option(
        None,
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
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Text prompt describing the requested edit",
        metavar="TEXT",
    ),
    negative_prompt: Optional[str] = typer.Option(
        None,
        "--negative-prompt",
        help="Text prompt describing what to avoid in the edited image",
        metavar="TEXT",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
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
        help="Optional seed for deterministic results. Same seed with same inputs yields repeatable outputs. Omit for random seed.",
        metavar="INT",
    ),
    strength: Optional[float] = typer.Option(
        None,
        "--strength",
        help="Edit strength in [0.0, 1.0]. Low (0.0–0.3)=subtle edits; Mid (0.4–0.7)=balanced; High (0.8–1.0)=strong changes, may introduce artifacts. Defaults to config if omitted.",
        min=0.0,
        max=1.0,
        metavar="FLOAT",
    ),
    guidance: Optional[float] = typer.Option(
        None,
        "--guidance",
        help="Classifier-free guidance scale (>= 0.0). Typical 1–20. Lower values favor creativity/diversity; higher values enforce the prompt more strictly (risk oversaturation). Defaults to config if omitted.",
        min=0.0,
        metavar="FLOAT",
    ),
    timeout: Optional[int] = typer.Option(
        None,
        "--timeout",
        help="HTTP timeout in seconds (>=1). Lower fails faster on slow networks; higher avoids timeouts but waits longer. Overrides config for this run.",
        min=1,
        metavar="SECONDS",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help=(
            "Explicit inference endpoint URL for Hugging Face; overrides provider/model for this run. "
            "Use this for backward compatibility or to target a specific HF endpoint."
        ),
        metavar="URL",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help=(
            "Inference provider to use (e.g., 'huggingface', 'replicate'). "
            "Defaults to settings.DEFAULT_PROVIDER (default: 'replicate' unless overridden in .env). "
            "Ignored if --endpoint is provided. Use '--providers' to list available providers."
        ),
        metavar="NAME",
    ),
    model: Optional[str] = typer.Option(
        "Qwen/Qwen-Image-Edit",
        "--model",
        help='Model repo id to use (default: "Qwen/Qwen-Image-Edit"). Ignored if --endpoint is provided.',
        metavar="REPO",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output with detailed error information",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode with full error context and artifact saving",
    ),
    # Providers listing option (replaces former 'providers' subcommand)
    providers: bool = typer.Option(
        False,
        "--providers",
        help="List available providers and exit",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="With --providers, output provider list as JSON",
    ),
) -> None:
    """
    Image editing CLI with a provider-agnostic architecture.

    Single-command usage:
      $ pdm run img-edit --input ./examples/assets/input.jpg \\
                         --prompt "Replace the sky with a sunset" \\
                         --output ./out/edited.jpg [other options]

    Provider selection:
      - Replicate (default): no flag needed
      - Hugging Face via provider: add --provider huggingface
      - Hugging Face via explicit endpoint: add --endpoint https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit

    Helpful ranges and effects:
      - --strength FLOAT [0.0–1.0]
          Low (0.0–0.3): subtle edits, preserves the original more
          Mid (0.4–0.7): balanced changes
          High (0.8–1.0): strong changes, higher chance of artifacts
      - --guidance FLOAT (>= 0.0, typical 1–20)
          Lower: more creative/diverse but weaker prompt adherence
          Higher: more literal prompt adherence, can reduce variety and oversaturate
      - --timeout SECONDS (>= 1)
          Lower: fail fast on slow responses
          Higher: avoid timeouts at the cost of waiting longer

    Provider discovery:
      - Use '--providers' to list available providers (add '--json' for machine-readable output)

    Notes:
      - The default provider is Replicate, but Hugging Face is fully supported.
      - Always invoke the image editing functionality via the single command: 'img-edit [OPTIONS]'.
    """
    # If provider listing requested, print it and exit early
    if providers:
        try:
            from lib.providers import list_providers, get_registry  # type: ignore
            from lib.config import Settings  # type: ignore
        except Exception as e:
            typer.secho(f"Failed to import provider registry: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)

        default_name = None
        try:
            default_name = Settings().DEFAULT_PROVIDER
        except Exception:
            default_name = None

        data = {}
        try:
            data = list_providers()  # type: ignore
        except Exception:
            try:
                reg = get_registry()  # type: ignore
                names = list(reg)
                data = {name: "" for name in names}
            except Exception:
                data = {}

        if json_output:
            out = {"default": default_name, "providers": data}
            typer.echo(json.dumps(out, indent=2))
            raise typer.Exit(code=0)

        if default_name:
            typer.echo(f"Default provider: {default_name}")
        if not data:
            typer.echo("No providers are currently registered.")
        else:
            typer.echo("Available providers:")
            for name, desc in data.items():
                mark = " (default)" if default_name and name == default_name else ""
                if desc:
                    typer.echo(f"  - {name}{mark}: {desc}")
                else:
                    typer.echo(f"  - {name}{mark}")
        raise typer.Exit(code=0)

    # Otherwise, perform the edit. Validate requireds here since options are optional on callback.
    if input_path is None or prompt is None or output_path is None:
        typer.secho(
            "Missing required options: --input, --prompt, --output\nRun 'pdm run img-edit --help' for usage.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    # Delegate to the edit command implementation for the actual work
    edit(
        input_path=input_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        output_path=output_path,
        mask=mask,
        seed=seed,
        strength=strength,
        guidance=guidance,
        timeout=timeout,
        endpoint=endpoint,
        provider=provider,
        model=model,
        verbose=verbose,
        debug=debug,
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


@app.command("edit", help="Edit an image using AI (default provider: replicate). See examples with '--help'.")
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
    negative_prompt: Optional[str] = typer.Option(
        None,
        "--negative-prompt",
        help="Text prompt describing what to avoid in the edited image",
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
        help="Optional seed for deterministic results. Same seed with same inputs yields repeatable outputs. Omit for random seed.",
        metavar="INT",
    ),
    strength: Optional[float] = typer.Option(
        None,
        "--strength",
        help="Edit strength in [0.0, 1.0]. Low (0.0–0.3)=subtle edits; Mid (0.4–0.7)=balanced; High (0.8–1.0)=strong changes, may introduce artifacts. Defaults to config if omitted.",
        min=0.0,
        max=1.0,
        metavar="FLOAT",
    ),
    guidance: Optional[float] = typer.Option(
        None,
        "--guidance",
        help="Classifier-free guidance scale (>= 0.0). Typical 1–20. Lower values favor creativity/diversity; higher values enforce the prompt more strictly (risk oversaturation). Defaults to config if omitted.",
        min=0.0,
        metavar="FLOAT",
    ),
    timeout: Optional[int] = typer.Option(
        None,
        "--timeout",
        help="HTTP timeout in seconds (>=1). Lower fails faster on slow networks; higher avoids timeouts but waits longer. Overrides config for this run.",
        min=1,
        metavar="SECONDS",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help=(
            "Explicit inference endpoint URL for Hugging Face; overrides provider/model for this run. "
            "Use this for backward compatibility or to target a specific HF endpoint."
        ),
        metavar="URL",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help=(
            "Inference provider to use (e.g., 'huggingface', 'replicate'). "
            "Defaults to settings.DEFAULT_PROVIDER (default: 'replicate' unless overridden in .env). "
            "Ignored if --endpoint is provided. Use '--providers' to list available providers."
        ),
        metavar="NAME",
    ),
    model: Optional[str] = typer.Option(
        "Qwen/Qwen-Image-Edit",
        "--model",
        help='Model repo id to use (default: "Qwen/Qwen-Image-Edit"). Ignored if --endpoint is provided.',
        metavar="REPO",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output with detailed error information",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode with full error context and artifact saving",
    ),
) -> None:
    """
    Edit an image with AI.

    Overview:
      - Default provider: Replicate (fast and cost-efficient)
      - Hugging Face is fully supported and can be selected with '--provider huggingface'
        or by setting an explicit Hugging Face '--endpoint' (which overrides provider/model)

    Required options:
      - --input / -i FILE   Path to the input image
      - --prompt / -p TEXT  Text instructions describing the edit
      - --output / -o FILE  Path to save the edited image

    Common options:
      - --provider NAME     'replicate' (default) or 'huggingface' (ignored if --endpoint is provided)
      - --endpoint URL      Explicit Hugging Face endpoint (overrides provider/model)
      - --strength FLOAT    Edit strength 0.0–1.0 (default from config: 0.8)
      - --guidance FLOAT    Guidance scale >= 0.0 (default from config: 7.5)
      - --seed INT          Deterministic seed
      - --negative-prompt TEXT  Text describing what to avoid in the edited image
      - --timeout SECONDS   Request timeout override in seconds

    Examples:
      1) Default provider (Replicate)
         pdm run img-edit edit -i ./examples/assets/input.jpg -p "Replace the sky with a sunset" -o ./out/edited.jpg

      2) With negative prompt to avoid specific elements
         pdm run img-edit edit -i input.jpg -p "Make it sunny" --negative-prompt "clouds, rain, gloomy" -o sunny.jpg

      3) Hugging Face by provider
         pdm run img-edit edit -i ./examples/assets/input.jpg -p "Cartoonize the photo" -o ./out/hf.jpg --provider huggingface

      4) Hugging Face with explicit endpoint (overrides provider/model)
         pdm run img-edit edit -i ./examples/assets/input.jpg -p "Make it black and white" -o ./out/mono.jpg \
           --endpoint https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit

    Behavior:
      - Ensures the output directory exists before invoking the pipeline
      - Constructs a Settings object honoring optional endpoint/timeout overrides
      - Invokes the pipeline's edit operation with provided parameters
      - Emits user-friendly messages and returns appropriate process exit codes

    Raises:
      - typer.Exit: With non-zero status if an error occurs
    """
    # Defer library imports to avoid config evaluation during --help
    try:
        from lib.config import Settings  # type: ignore
        from lib.edit_pipeline import ImageEditPipeline  # type: ignore
        from lib.logging_utils import get_logger, setup_logging  # type: ignore
        from lib.errors import format_error_for_cli, save_error_artifact  # type: ignore
    except Exception as e:
        typer.secho(f"Failed to import application modules: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    # Configure logging based on verbosity
    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    setup_logging(level=log_level, filename="logs/imgedit.log" if debug else None)
    
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
        # Endpoint wins over provider/model if provided
        settings_overrides["HF_INFERENCE_ENDPOINT"] = endpoint
    if model is not None:
        settings_overrides["IMG_EDIT_MODEL"] = model

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

    # Determine and validate the effective provider selection
    # If endpoint is provided, prefer 'huggingface' provider since endpoint applies there
    try:
        from lib.providers import get_provider_class, list_providers, get_registry  # type: ignore
    except Exception:
        get_provider_class = None  # type: ignore
        list_providers = None  # type: ignore
        get_registry = None  # type: ignore

    effective_provider = None
    if endpoint is not None:
        effective_provider = "huggingface"
    elif provider is not None:
        effective_provider = provider

    selected_provider = effective_provider or getattr(settings, "DEFAULT_PROVIDER", None)

    # Validate provider name against provider registry if possible
    if selected_provider and get_provider_class is not None:
        try:
            get_provider_class(selected_provider)  # may raise KeyError if not registered
        except KeyError as e:
            available = []
            try:
                if list_providers is not None:
                    mapping = list_providers()
                    available = sorted(mapping.keys()) if mapping else []
            except Exception:
                try:
                    if get_registry is not None:
                        registry = get_registry()
                        available = sorted(list(registry))
                except Exception:
                    available = []
            typer.secho(
                f"Invalid provider '{selected_provider}'. Available providers: {', '.join(available) if available else 'none'}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        except Exception:
            # If provider registry cannot be consulted, continue; pipeline will surface errors
            pass

    # Apply the effective provider to settings so the pipeline uses it
    if selected_provider:
        try:
            settings.DEFAULT_PROVIDER = selected_provider  # type: ignore[attr-defined]
            # Keep legacy field in sync for downstream compatibility
            if hasattr(settings, "IMG_EDIT_PROVIDER"):
                settings.IMG_EDIT_PROVIDER = selected_provider  # type: ignore[attr-defined]
        except Exception:
            # Continue even if assignment fails; pipeline may still resolve a provider
            pass

    # Prepare extra payload args
    extra_inputs = {}
    try:
        if mask is not None:
            # Read and encode mask as base64 string so the HF client can include it in JSON payload
            extra_inputs["mask"] = _encode_file_b64(mask)
        if negative_prompt is not None:
            # Add negative prompt to extra inputs
            extra_inputs["negative_prompt"] = negative_prompt
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

            # Summarize provider/model and parameters used
            eff_provider = selected_provider or getattr(settings, "DEFAULT_PROVIDER", None) or "unknown"
            provider_note = "(default)" if (provider is None and endpoint is None) else ""
            # Determine model used
            if endpoint is not None or eff_provider == "huggingface":
                eff_model = getattr(settings, "IMG_EDIT_MODEL", "Qwen/Qwen-Image-Edit")
                model_note = "(default)" if (model == "Qwen/Qwen-Image-Edit") else "(overridden)"
            elif eff_provider == "replicate":
                eff_model = "qwen/qwen-image-edit"
                model_note = "(provider default)"
            else:
                eff_model = getattr(settings, "IMG_EDIT_MODEL", "Qwen/Qwen-Image-Edit")
                model_note = ""
            # Effective parameter values
            used_strength = strength if strength is not None else getattr(settings, "IMG_EDIT_DEFAULT_STRENGTH", 0.8)
            used_guidance = guidance if guidance is not None else getattr(settings, "IMG_EDIT_DEFAULT_GUIDANCE", 7.5)
            used_seed = seed if seed is not None else None  # None => random
            used_timeout = getattr(settings, "IMG_EDIT_TIMEOUT_SECONDS", 120)
            mask_used = mask is not None
            negative_prompt_used = negative_prompt is not None
            # Default markers
            strength_note = "(default)" if strength is None else "(user)"
            guidance_note = "(default)" if guidance is None else "(user)"
            timeout_note = "(default)" if timeout is None else "(overridden)"
            seed_display = str(used_seed) if used_seed is not None else "random (unset)"

            typer.echo("Run summary:")
            typer.echo(f"  Provider: {eff_provider} {provider_note}")
            if endpoint:
                typer.echo(f"  Endpoint: {endpoint}")
            typer.echo(f"  Model: {eff_model} {model_note}")
            typer.echo("  Parameters:")
            typer.echo(f"    prompt: {prompt!r}")
            typer.echo(f"    mask: {'yes' if mask_used else 'no'}")
            typer.echo(f"    negative_prompt: {'yes' if negative_prompt_used else 'no'}")
            typer.echo(f"    strength: {used_strength} {strength_note}")
            typer.echo(f"    guidance: {used_guidance} {guidance_note}")
            typer.echo(f"    seed: {seed_display}")
            typer.echo(f"    timeout: {used_timeout}s {timeout_note}")
            
            # Save success artifact in debug mode
            if debug:
                # Use model_dump() for Pydantic v2, fallback to dict() or vars()
                settings_dict = {}
                if hasattr(settings, 'model_dump'):
                    settings_dict = settings.model_dump()
                elif hasattr(settings, 'dict'):
                    settings_dict = settings.dict()
                else:
                    settings_dict = vars(settings)
                
                # Also persist a run summary for diagnostics
                run_summary = {
                    "provider": eff_provider,
                    "provider_default": (provider is None and endpoint is None),
                    "model": eff_model,
                    "endpoint": endpoint,
                    "parameters": {
                        "prompt": prompt,
                        "negative_prompt_provided": negative_prompt_used,
                        "mask_provided": mask_used,
                        "strength": used_strength,
                        "strength_default": strength is None,
                        "guidance": used_guidance,
                        "guidance_default": guidance is None,
                        "seed": used_seed,
                        "seed_random": used_seed is None,
                        "timeout_seconds": used_timeout,
                        "timeout_default": timeout is None,
                    },
                }
                success_artifact = {
                    "status": "success",
                    "result": result,
                    "settings": settings_dict,
                    "run_summary": run_summary,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                os.makedirs("tmp", exist_ok=True)
                with open("tmp/last_success.json", "w") as f:
                    json.dump(success_artifact, f, indent=2)
                logger.debug("Success artifact saved to tmp/last_success.json")
                
            raise typer.Exit(code=0)
        else:
            # Error result from pipeline
            error_message = format_error_for_cli(Exception(message), verbose or debug)
            typer.secho(f"Edit failed: {error_message}", fg=typer.colors.RED, err=True)

            # Print run summary to aid diagnostics
            eff_provider = selected_provider or getattr(settings, "DEFAULT_PROVIDER", None) or "unknown"
            provider_note = "(default)" if (provider is None and endpoint is None) else ""
            if endpoint is not None or eff_provider == "huggingface":
                eff_model = getattr(settings, "IMG_EDIT_MODEL", "Qwen/Qwen-Image-Edit")
                model_note = "(default)" if (model == "Qwen/Qwen-Image-Edit") else "(overridden)"
            elif eff_provider == "replicate":
                eff_model = "qwen/qwen-image-edit"
                model_note = "(provider default)"
            else:
                eff_model = getattr(settings, "IMG_EDIT_MODEL", "Qwen/Qwen-Image-Edit")
                model_note = ""
            used_strength = strength if strength is not None else getattr(settings, "IMG_EDIT_DEFAULT_STRENGTH", 0.8)
            used_guidance = guidance if guidance is not None else getattr(settings, "IMG_EDIT_DEFAULT_GUIDANCE", 7.5)
            used_seed = seed if seed is not None else None
            used_timeout = getattr(settings, "IMG_EDIT_TIMEOUT_SECONDS", 120)
            mask_used = mask is not None
            negative_prompt_used = negative_prompt is not None
            strength_note = "(default)" if strength is None else "(user)"
            guidance_note = "(default)" if guidance is None else "(user)"
            timeout_note = "(default)" if timeout is None else "(overridden)"
            seed_display = str(used_seed) if used_seed is not None else "random (unset)"

            typer.echo("Run summary:", err=True)
            typer.echo(f"  Provider: {eff_provider} {provider_note}", err=True)
            if endpoint:
                typer.echo(f"  Endpoint: {endpoint}", err=True)
            typer.echo(f"  Model: {eff_model} {model_note}", err=True)
            typer.echo("  Parameters:", err=True)
            typer.echo(f"    prompt: {prompt!r}", err=True)
            typer.echo(f"    mask: {'yes' if mask_used else 'no'}", err=True)
            typer.echo(f"    negative_prompt: {'yes' if negative_prompt_used else 'no'}", err=True)
            typer.echo(f"    strength: {used_strength} {strength_note}", err=True)
            typer.echo(f"    guidance: {used_guidance} {guidance_note}", err=True)
            typer.echo(f"    seed: {seed_display}", err=True)
            typer.echo(f"    timeout: {used_timeout}s {timeout_note}", err=True)
            
            # Save error artifact in debug mode or if verbose
            if debug or verbose:
                # Use model_dump() for Pydantic v2, fallback to dict() or vars()
                settings_dict = {}
                if hasattr(settings, 'model_dump'):
                    settings_dict = settings.model_dump()
                elif hasattr(settings, 'dict'):
                    settings_dict = settings.dict()
                else:
                    settings_dict = vars(settings)
                
                # Also persist run summary for diagnostics
                run_summary = {
                    "provider": eff_provider,
                    "provider_default": (provider is None and endpoint is None),
                    "model": eff_model,
                    "endpoint": endpoint,
                    "parameters": {
                        "prompt": prompt,
                        "negative_prompt_provided": negative_prompt_used,
                        "mask_provided": mask_used,
                        "strength": used_strength,
                        "strength_default": strength is None,
                        "guidance": used_guidance,
                        "guidance_default": guidance is None,
                        "seed": used_seed,
                        "seed_random": used_seed is None,
                        "timeout_seconds": used_timeout,
                        "timeout_default": timeout is None,
                    },
                }
                error_artifact = result.copy()
                error_artifact["settings"] = settings_dict
                error_artifact["run_summary"] = run_summary
                error_artifact["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                os.makedirs("tmp", exist_ok=True)
                with open("tmp/last_error.json", "w") as f:
                    json.dump(error_artifact, f, indent=2)
                logger.debug("Error artifact saved to tmp/last_error.json")
                
            raise typer.Exit(code=1)

    except typer.Exit:
        # Allow Typer to handle explicit exit codes without additional formatting
        raise
    except Exception as e:
        # Format error based on verbosity - use our error formatting function
        from lib.errors import format_error_for_cli
        error_message = format_error_for_cli(e, verbose or debug)
        typer.secho(f"{error_message}", fg=typer.colors.RED, err=True)
        
        # Save error artifact in debug mode
        if debug:
            try:
                from lib.errors import save_error_artifact
                save_error_artifact(e, "tmp/last_error.json")
                logger.debug("Error artifact saved to tmp/last_error.json")
            except Exception as save_error:
                logger.warning(f"Failed to save error artifact: {save_error}")
        
        raise typer.Exit(code=1)


# Note: Previous 'providers' subcommand has been removed.
# Use the global '--providers' option instead:
#   pdm run img-edit --providers [--json]


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