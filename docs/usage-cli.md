# CLI Usage

The CLI is available via the PDM script "img-edit" defined in [pyproject.toml](../pyproject.toml). Implementation lives in [src/cli/main.py](../src/cli/main.py) and uses shared logic from [src/lib](../src/lib).

## Synopsis (no subcommand):
```bash
pdm run img-edit \
  --input ./examples/assets/input.jpg \
  --prompt "Replace the sky with a sunset" \
  --output ./极速赛车开奖结果历史记录out/edited.jpg \
  [--mask ./examples/assets/mask.png] \
  [--seed 42] \
  [--strength 0.8] \
  [--guidance 7.5] \
  [--timeout 120] \
  [--provider fal-ai] \
  [--model Qwen/Qwen-Image-Edit] \
  [--endpoint https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit] \
  [--verbose] \
  [--debug]
```

## Short options:
- -i / --input FILE
- -p / --prompt TEXT
- -o / --output FILE
- -m / --mask FILE
- --seed INT
- --strength FLOAT (0.0–1.0)
- --guidance FLOAT (>= 0.0)
- --timeout SECONDS (>= 1)
- --provider NAME (default: fal-ai)
- --model REP极速赛车开奖结果历史记录O (default: Qwen/Qwen-Image-Edit)
- --endpoint URL (if provided, overrides provider/model)
- -v / --verbose: Enable verbose output with detailed error messages
- -d / --debug: Enable debug mode with full error context and artifact saving

## Help:
```bash
pdm run img-edit --help
```

## Error Handling and Diagnostics

The CLI now includes comprehensive error handling with rich context and user guidance:

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
When using `--debug`, the following artifacts are saved:
- `tmp/last_error.json`: Structured error information with full context
- `tmp/last_error.txt`: Human-readable error summary
- `tmp/last_success.json`: Success operation details

### Exit Codes:
- 0: Success
- 1: Operational failure (validation, IO, API/network errors)
- 2: Configuration/import error (missing tokens, module issues)

## Notes:
- Run commands from the project root so relative paths resolve correctly
- Token environment variables: HF_TOKEN (preferred) or HF_API_TOKEN (fallback); see [docs/setup.md](./setup.md)
- You can override provider/model/endpoint per-run; if --endpoint is provided it takes precedence over provider/model
- For detailed error information and troubleshooting, see [docs/errors.md](./errors.md)

## Examples:

**Basic usage:**
```bash
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses"
```

**Verbose output:**
```bash
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses" --verbose
```

**Debug mode with full diagnostics:**
```bash
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses" --debug
```

**Custom provider and timeout:**
```bash
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses" --provider fal-ai --timeout 180