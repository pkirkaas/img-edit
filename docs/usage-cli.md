# CLI Usage

The CLI is available via the PDM script "img-edit" defined in [pyproject.toml](../pyproject.toml). Implementation lives in [src/cli/main.py](../src/cli/main.py) and uses shared logic from [src/lib](../src/lib).

Synopsis (no subcommand):
```bash
pdm run img-edit \
  --input ./examples/assets/input.jpg \
  --prompt "Replace the sky with a sunset" \
  --output ./out/edited.jpg \
  [--mask ./examples/assets/mask.png] \
  [--seed 42] \
  [--strength 0.8] \
  [--guidance 7.5] \
  [--timeout 120] \
  [--provider fal-ai] \
  [--model Qwen/Qwen-Image-Edit] \
  [--endpoint https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit]
```

Short options:
- -i / --input FILE
- -p / --prompt TEXT
- -o / --output FILE
- -m / --mask FILE
- --seed INT
- --strength FLOAT (0.0–1.0)
- --guidance FLOAT (>= 0.0)
- --timeout SECONDS (>= 1)
- --provider NAME (default: fal-ai)
- --model REPO (default: Qwen/Qwen-Image-Edit)
- --endpoint URL (if provided, overrides provider/model)

Help:
```bash
pdm run img-edit --help
```

Notes:
- Run commands from the project root so relative paths resolve correctly
- Token environment variables: HF_TOKEN (preferred) or HF_API_TOKEN (fallback); see [docs/setup.md](./setup.md)
- You can override provider/model/endpoint per-run; if --endpoint is provided it takes precedence over provider/model

Exit codes:
- 0: success
- 1: operational failure (validation, IO, API/network)
- 2: configuration/import error (e.g., missing HF_TOKEN/HF_API_TOKEN)