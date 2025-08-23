# AI Image Edit

An AI powered image editor using the Qwen/Qwen-Image-Edit model via the Hugging Face Serverless Inference API. Provides a Typer-based CLI with a future GUI.

- Implemented in Python
- Initially based on Qwen/Qwen-Image-Edit
- CLI & GUI

## Quickstart

1) Install and set up
- Ensure PDM is installed; see [docs/setup.md](docs/setup.md) for environment setup and .env configuration
- Install dependencies:
```bash
pdm install
```
- Add your own images to [examples/assets/](examples/assets/) (see [examples/assets/README.md](examples/assets/README.md)). The repository does not include binary images.

2) Verify the CLI
```bash
pdm run img-edit --help
```

3) Edit an image (example)
```bash
pdm run img-edit edit \
  --input ./examples/assets/input.jpg \
  --prompt "Replace the sky with a sunset" \
  --output ./out/edited.jpg
```
See the full CLI reference in [docs/usage-cli.md](docs/usage-cli.md). Run all commands from the project root.

## Documentation

- Setup and configuration: [docs/setup.md](docs/setup.md)
- CLI usage and options: [docs/usage-cli.md](docs/usage-cli.md)
- Examples: [examples/README.md](examples/README.md)

Note: The CLI entry point is defined in [pyproject.toml](pyproject.toml) as the PDM script "img-edit". The implementation lives in [src/cli/main.py](src/cli/main.py).