# Architecture Overview

Layers (planned):

- IO: [src/lib/image_io.py](src/lib/image_io.py) — read/write images, masks, format validation
- Config: [src/lib/config.py](src/lib/config.py) — environment-backed settings (tokens, endpoints, timeouts)
- Client: [src/lib/hf_client.py](src/lib/hf_client.py) — HTTP calls to Hugging Face Serverless Inference
- Pipeline: [src/lib/edit_pipeline.py](src/lib/edit_pipeline.py) — orchestrates request payloads and response decoding
- CLI: [src/cli/main.py](src/cli/main.py) — Typer-based CLI, progress and rich output
- Logging: [src/lib/logging_utils.py](src/lib/logging_utils.py) — structured logging utilities

Data flow:

Image/Mask files -> [src/lib/image_io.py](src/lib/image_io.py) -> [src/lib/config.py](src/lib/config.py) -> [src/lib/hf_client.py](src/lib/hf_client.py) -> [src/lib/edit_pipeline.py](src/lib/edit_pipeline.py) -> Output image via [src/lib/image_io.py](src/lib/image_io.py)

Documentation governance: see [.roo/rules/running-docs.md](.roo/rules/running-docs.md) — docs must be kept up to date with implementation changes.