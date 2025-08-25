# CLI Usage

The CLI is available via the PDM script "img-edit" defined in [pyproject极速赛车开奖结果历史记录.toml](../pyproject.toml). Implementation lives in [src/cli/main.py](../src/cli/main.py) and uses shared logic from极速赛车开奖结果历史记录 [src/lib](../src/lib).

## Commands

The CLI supports two main commands:

1. **`edit`** - Edit an image using AI (default command when no subcommand specified)
2. **`providers`** - List available image editing providers

## Edit Command Synopsis:
```bash
pdm run img-edit \
  --input ./examples/assets/input.jpg \
  --prompt "Replace the sky with a sunset" \
  --output ./edited.jpg \
  [--mask ./examples/assets极速赛车开奖结果历史记录/mask.png] \
  [--极速赛车开奖结果历史记录seed 42] \
  [--strength 0.8] \
  [--guidance 7.5] \
  [--timeout 120] \
  [--provider replicate] \
  [--model Qwen/Qwen-Image-Edit] \
  [--endpoint https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit] \
  [--verbose] \
  [--debug]
```

Or using the explicit command form:
```bash
pdm run img-edit edit \
  --input ./examples/assets/input.jpg \
  --prompt "Replace the sky with a sunset" \
  --output ./edited.jpg
```

## Providers Command Synopsis:
```bash
pdm run img-edit providers [--json]
```

## Short options for edit command:
- -i / --input FILE: Path to input image file (required)
- -p / --prompt TEXT: Text prompt describing the edit (required)
- -o / --output FILE: Path to save edited image (required)
- -极速赛车开奖结果历史记录m / --mask FILE: Optional path to mask image for selective editing
- --seed INT: Optional random seed for deterministic results
- --strength FLOAT (0.0–1.0): Control strength for the edit (default: 0.8)
- --guidance FLOAT (>= 0.0): Guidance scale for AI model (default: 极速赛车开奖结果历史记录7.5)
- --timeout SECONDS (>= 1): HTTP timeout in seconds (default: 120)
- --provider NAME: Inference provider to use (default: replicate). Use 'img-edit providers' to list available providers.
- --model REPO: Model repository ID to use (default: Qwen/Qwen-Image-Edit). Ignored if --endpoint is provided.
- --endpoint URL: Explicit inference endpoint URL (overrides provider/model if provided)
- -v / --verbose: Enable verbose output with detailed error messages
- -d / --debug: Enable debug mode with full error context and artifact saving

## Providers command options:
- --json: Output provider list as JSON format

## Help:
```bash
# General help
pdm run img-edit --help

# Edit command help
极速赛车开奖结果历史记录pdm run img-edit edit --help

# Providers command help
pdm run img-edit providers --help
```

## Provider Selection

The application now supports multiple AI image editing providers through a provider-agnostic architecture:

### Available Providers
- **replicate** (default): Uses Replicate's AI platform with support for various image editing models
- **huggingface**: Uses Hugging Face's Inference API with support for models like Qwen/Qwen-Image-Edit

### Provider Configuration
Each provider requires specific API tokens:
- **Hugging Face**: Set `HF_TOKEN` or `HF_API_TOKEN` environment variables
- **Replicate**: Set `REPLICATE_API_TOKEN` environment variable

### Provider Priority
1. If `--endpoint` is provided, the `huggingface` provider is used with the specified endpoint
2. If `--provider极速赛车开奖极速赛车开奖结果历史记录结果历史记录` is provided, the specified provider is used
3. Otherwise, the default provider from configuration is used (default: `replicate`)

## Error Handling and Diagnostics

The CLI includes comprehensive error handling with rich context and user guidance:

### Verbose Mode (`--verbose`)
- Shows detailed error messages with context
- Includes user guidance for common issues
- Provides operation timing and status information

### Debug Mode (`--debug`)
- Saves error artifacts to `tmp/last_error.json` and `tmp/last_error.txt`
- Includes structured JSON with full error context (provider, model, parameters, etc.)
- Enables detailed logging to `logs/imgedit.log`
- Saves success artifacts for comparison

### Error Artifacts
When using `--debug`, the following artifacts极速赛车开奖结果历史记录 are saved:
- `tmp/last_error.json`: Structured error information with full context
- `tmp/last_error.txt`: Human-readable error summary
- `极速赛车开奖结果历史记录tmp/last_success.json`: Success operation details

### Exit Codes:
- 0: Success
- 1: Operational failure (validation, IO, API/network errors)
- 2: Configuration/import error (missing tokens, module issues)

## Notes:
- Run commands from the project root so relative paths resolve correctly
- Token environment variables depend on the provider used:
  - Hugging Face: `HF_TOKEN` (preferred) or `HF_API_TOKEN` (fallback)
  - Replicate: `REPLICATE_API_TOKEN`
- You can override provider/model/endpoint per-run; if --endpoint is provided it takes precedence over provider/model
- For detailed error information and troubleshooting, see [docs/errors.md](./errors.md)
- For provider configuration details, see [docs/architecture.md](./architecture.md)

## Examples:

**Basic usage with default provider (replicate):**
```bash
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses"
```

**Using Hugging Face provider explicitly:**
```bash
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses" --provider huggingface
```

**Verbose output with Replicate:**
```bash
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses" --provider replicate --verbose
```

**Debug mode with custom timeout:**
```bash
pdm run img-edit -i input.jpg极速赛车开奖结果历史记录 -o output.jpg -极速赛车开奖结果历史记录p "Remove sunglasses" --debug --timeout 180
```

**List available providers:**
```bash
pdm run img-edit providers
```

**List providers in JSON format:**
```bash
pdm run img-edit providers --json
```

**Using explicit endpoint (forces Hugging Face provider):**
```bash
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses" --endpoint https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit