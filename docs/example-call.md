# Example Python code for Qwen Image Edit via huggingface_hub.InferenceClient

This mirrors our implementation using a provider-based configuration. Token preference is HF_TOKEN (preferred) with fallback to HF_API_TOKEN as documented in [docs/setup.md](setup.md). See the thin client wrapper in [src/lib/hf_client.py](../src/lib/hf_client.py) and its usage in the pipeline [src/lib/edit_pipeline.py](../src/lib/edit_pipeline.py).

## Example Code
```py
import os
from huggingface_hub import InferenceClient

# Prefer HF_TOKEN; if absent, you can also set HF_API_TOKEN
client = InferenceClient(
    provider="fal-ai",
    api_key=os.environ["HF_TOKEN"],
)

with open("cat.png", "rb") as image_file:
    input_image = image_file.read()

# Output is a PIL.Image.Image
image = client.image_to_image(
    input_image,
    prompt="Turn the cat into a tiger.",
    model="Qwen/Qwen-Image-Edit",
    # Optional parameters:
    # strength=0.8,
    # guidance_scale=7.5,
    # seed=42,
)
```