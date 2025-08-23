# Setup

This project uses PDM for dependency and environment management; Python target is >=3.13 per [pyproject.toml](pyproject.toml).

## Prerequisites
- Python 3.13 or newer available on PATH
- PDM installed:
  - Windows (Cygwin Bash/WSL/PowerShell): `pip install -U pdm`

## Install
```bash
pdm install
```

## Environment variables
Copy [.env.example](.env.example) to `.env` and set your token:
```bash
cp .env.example .env
# edit .env to set HF_API_TOKEN
```

Required:
- HF_API_TOKEN: Your Hugging Face token for serverless inference

Defaults provided:
- HF_INFERENCE_ENDPOINT: Qwen/Qwen-Image-Edit model endpoint

## Quick check
After implementing the CLI in [src/cli/main.py](src/cli/main.py), you will be able to run:
```bash
pdm run img-edit --help
```

See [docs/usage-cli.md](docs/usage-cli.md) for usage once the CLI is available.