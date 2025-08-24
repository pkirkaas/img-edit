# Setup

This project uses PDM for dependency and environment management; Python target is >=3.13 per [pyproject.toml](../pyproject.toml).

## Prerequisites
- Python 3.13 or newer available on PATH
- PDM installed:
  - Windows (Cygwin Bash/WSL/PowerShell): `pip install -U pdm`

## Install
```bash
pdm install
```

## Environment variables
Copy [.env.example](../.env.example) to `.env` and set your token:
```bash
cp .env.example .env
# edit .env to set HF_TOKEN (preferred) or HF_API_TOKEN (fallback)
```

Required:
- HF_TOKEN (preferred) or HF_API_TOKEN (fallback): Your Hugging Face token for InferenceClient

Optional defaults:
- IMG_EDIT_PROVIDER: defaults to "fal-ai"
- IMG_EDIT_MODEL: defaults to "Qwen/Qwen-Image-Edit"
- HF_INFERENCE_ENDPOINT: optional explicit endpoint; if set, it overrides provider/model for the run

## Quick check
Run the CLI help to verify imports and configuration load cleanly:
```bash
pdm run img-edit --help
```

See [usage-cli.md](usage-cli.md) for the full command reference. Implementation lives in [src/cli/main.py](../src/cli/main.py).