# CLI Usage

The CLI is available via the PDM script "img-edit" defined in [`pyproject.toml`](../pyproject.toml). Implementation lives in [`src/cli/main.py`](../src/cli/main.py) and uses shared logic from [`src/lib`](../src/lib).

## Single-Command Interface

The CLI now uses a single-command interface. All invocations are treated as an "edit" operation by default. Use the `--providers` option to list available providers.

### Synopsis
```bash
pdm run img-edit \
  --input ./examples/assets/input.jpg \
  --prompt "Replace the sky with a sunset" \
  --output ./edited.jpg \
  [--mask ./examples/assets/mask.png] \
  [--seed 42] \
  [--strength 0.8] \
  [--guidance 7.5] \
  [--timeout 120] \
  [--provider replicate|huggingface] \
  [--model Qwen/Qwen-Image-Edit] \
  [--endpoint https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit] \
  [--verbose] \
  [--debug]
```

You may still explicitly use the legacy subcommand for backward compatibility:
```bash
pdm run img-edit edit --input ... --prompt ... --output ...
```

### Providers Listing
```bash
# Human-readable
pdm run img-edit --providers

# JSON output
pdm run img-edit --providers --json
```

## Options

- -i / --input FILE
  - Required. Path to input image file.
- -p / --prompt TEXT
  - Required. Text prompt describing the requested edit.
- -o / --output FILE
  - Required. Path to save the edited image.
- -m / --mask FILE
  - Optional path to a mask image file for selective editing. The mask is sent as base64 in the request.
- --seed INT
  - Optional seed for deterministic results. Using the same seed and inputs yields repeatable outputs. Omit for random seed.
- --strength FLOAT
  - Range: [0.0, 1.0]
  - Meaning:
    - Low (0.0–0.3): Subtle edits, preserves the original more
    - Mid (0.4–0.7): Balanced changes
    - High (0.8–1.0): Strong changes, higher chance of artifacts
  - Default: From configuration (typically 0.8)
- --guidance FLOAT
  - Range: [0.0, ∞), typical effective range ~1–20
  - Meaning:
    - Lower values: More creative/diverse, weaker prompt adherence
    - Higher values: Strong prompt adherence, can reduce variety and oversaturate
  - Default: From configuration (typically 7.5)
- --timeout SECONDS
  - Range: [1, ∞)
  - Meaning:
    - Lower: Fails faster on slow networks or heavy loads
    - Higher: Avoids timeouts at the cost of waiting longer
  - Default: From configuration (typically 120 seconds)
- --provider NAME
  - Inference provider to use. Values: `replicate` (default), `huggingface`.
  - Ignored if `--endpoint` is provided.
  - Use `--providers` to list available providers.
- --model REPO
  - Model repository ID to use, e.g. `Qwen/Qwen-Image-Edit`. Ignored if `--endpoint` is provided.
- --endpoint URL
  - Explicit Hugging Face endpoint URL; overrides provider/model for this run (forces Hugging Face provider).
- -v / --verbose
  - Enable verbose output with detailed error information.
- -d / --debug
  - Enable debug mode with full error context, JSON artifacts, and detailed logging.
- --providers
  - List available providers and exit.
- --json
  - When used with `--providers`, output the provider list as JSON.

## Help
```bash
# General help
pdm run img-edit --help
```

## Provider Selection

The application supports multiple AI image editing providers through a provider-agnostic architecture.

### Available Providers
- replicate (default): Uses Replicate's AI platform with support for various image editing models
- huggingface: Uses Hugging Face's Inference API with support for models like Qwen/Qwen-Image-Edit

### Provider Configuration
Each provider requires specific API tokens:
- Hugging Face: Set `HF_TOKEN` or `HF_API_TOKEN` environment variables
- Replicate: Set `REPLICATE_API_TOKEN` environment variable

### Provider Priority
1. If `--endpoint` is provided, the `huggingface` provider is used with the specified endpoint
2. If `--provider` is provided, the specified provider is used
3. Otherwise, the default provider from configuration is used (default: `replicate`)

## Error Handling and Diagnostics

The CLI includes comprehensive error handling with rich context and user guidance.

### Verbose Mode (`--verbose`)
- Shows detailed error messages with context
- Includes user guidance for common issues
- Provides operation timing and status information

### Debug Mode (`--debug`)
- Saves error artifacts to `tmp/last_error.json`
- Includes structured JSON with full error context (provider, model, parameters, etc.)
- Enables detailed logging to `logs/imgedit.log`
- Saves success artifacts for comparison (`tmp/last_success.json`)

### Exit Codes
- 0: Success
- 1: Operational failure (validation, IO, API/network errors)
- 2: Configuration/import error (missing tokens, module issues)

## Notes
- Run commands from the project root so relative paths resolve correctly
- Token environment variables depend on the provider used:
  - Hugging Face: `HF_TOKEN` (preferred) or `HF_API_TOKEN` (fallback)
  - Replicate: `REPLICATE_API_TOKEN`
- You can override provider/model/endpoint per-run; if `--endpoint` is provided it takes precedence over provider/model
- For detailed error information and troubleshooting, see [`docs/errors.md`](./errors.md)
- For provider configuration details, see [`docs/architecture.md`](./architecture.md)

## Examples

Basic usage with default provider (replicate):
```bash
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses"
```

Using Hugging Face provider explicitly:
```bash
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses" --provider huggingface
```

Verbose output with Replicate:
```bash
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses" --provider replicate --verbose
```

Debug mode with custom timeout:
```bash
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses" --debug --timeout 180
```

List available providers:
```bash
pdm run img-edit --providers
```

List providers in JSON format:
```bash
pdm run img-edit --providers --json
```

Using explicit endpoint (forces Hugging Face provider):
```bash
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses" --endpoint https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit