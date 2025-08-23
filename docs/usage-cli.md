# CLI Usage (planned)

The CLI will expose an `edit` command to send an image and prompt to the Hugging Face serverless inference endpoint for Qwen Image Edit.

Planned command:
```bash
pdm run img-edit edit \
  --input ./examples/assets/input.jpg \
  --prompt "Replace the sky with a sunset" \
  --output ./out/edited.jpg \
  [--mask ./examples/assets/mask.png] \
  [--seed 42] \
  [--strength 0.8] \
  [--guidance 7.5] \
  [--timeout 120] \
  [--endpoint https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit]
```

This command will be implemented in [src/cli/main.py](src/cli/main.py) and will use shared logic from [src/lib](src/lib). Configuration will be read from environment variables loaded via `.env` (see [docs/setup.md](docs/setup.md)).