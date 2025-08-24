# Examples and Sample Usage

This directory provides runnable examples for the Typer-based CLI implemented in [src/cli/main.py](../src/cli/main.py) and exposed via PDM as configured in [pyproject.toml](../pyproject.toml). Use these examples to quickly test prompting, masking, and parameter overrides with your own images.

Before running, complete environment setup per [docs/setup.md](../docs/setup.md). For the full CLI command reference, see [docs/usage-cli.md](../docs/usage-cli.md).

## Add your sample images

No binary assets are stored in this repository. Add your own images to [examples/assets/](assets/) following the guidance in [examples/assets/README.md](assets/README.md).

Recommendations:
- Formats: JPEG (.jpg/.jpeg) or PNG (.png)
- Size: short side ~512–1024 px for faster runs and good quality
- Masks: PNG with same width/height as input; binary black/white or grayscale works well

Suggested filenames used below:
- Input image: ./examples/assets/input.jpg
- Mask image: ./examples/assets/mask.png

Run all commands from the project root directory.

## Quick help

```bash
pdm run img-edit --help
```

## Example 1 — Basic image edit with prompt

```bash
pdm run img-edit \
  --input ./examples/assets/input.jpg \
  --prompt "Replace the sky with a sunset" \
  --output ./out/edited.jpg
```

Notes:
- The output directory (./out) is created automatically if it does not exist
- Authentication and defaults are loaded from your .env as described in [docs/setup.md](../docs/setup.md)

## Example 2 — Edit with mask for precise control

```bash
pdm run img-edit \
  -i ./examples/assets/input.jpg \
  -p "Change the background to pure white, keep the subject unchanged" \
  -m ./examples/assets/mask.png \
  -o ./out/edited_masked.png
```

Notes:
- The mask is Base64-encoded and sent to the API
- Ensure the mask dimensions match the input image dimensions

## Example 3 — Using parameters (strength, guidance, seed)

```bash
pdm run img-edit \
  -i ./examples/assets/input.jpg \
  -p "Transform the photo into a soft watercolor painting" \
  -o ./out/edited_params.jpg \
  --strength 0.7 \
  --guidance 8.0 \
  --seed 42
```

Parameter tips:
- strength (0.0–1.0): how strongly the edit is applied (defaults from config)
- guidance (> 0): higher adheres more to prompt (defaults from config)
- seed: set to reproduce results deterministically

## Optional overrides — timeout and endpoint

You can override the HTTP timeout and the Hugging Face endpoint for a single run:

```bash
pdm run img-edit \
  -i ./examples/assets/input.jpg \
  -p "Add cinematic golden-hour lighting" \
  -o ./out/edited_custom.jpg \
  --timeout 180 \
  --endpoint https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit
```

## Troubleshooting

- Configuration error mentioning HF_API_TOKEN:
  - Review and follow [docs/setup.md](../docs/setup.md) to populate your [.env](../.env.example)
- Command layout and planned options:
  - Refer to [docs/usage-cli.md](../docs/usage-cli.md) for the canonical command structure
- Implementation details:
  - See the CLI entry point in [src/cli/main.py](../src/cli/main.py) and the supporting library modules in [src/lib](../src/lib)
