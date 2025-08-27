# Example Code for Image Editing with Multiple Providers

This document provides examples of how to use the image editing application with different AI providers. The application now supports a provider-agnostic architecture, allowing you to choose between Hugging Face and Replicate as your AI service provider.

## Provider-Agnostic Usage

The application now uses a provider abstraction layer, making it easy to switch between different AI services. Here's how to use it with different providers:

### Using the Default Provider (Replicate)

```bash
# Basic usage with default Replicate provider
pdm run img-edit \
  --input input.jpg \
  --prompt "Make the cat wear a party hat" \
  --output output.jpg

# With custom parameters
pdm run img-edit \
  --input input.jpg \
  --prompt "Make the cat wear a party hat" \
  --output output.jpg \
  --strength 0.7 \
  --guidance 8.0 \
  --seed 42 \
  --timeout 180
```

### Using Hugging Face Provider Explicitly

```bash
# Explicitly specify Hugging Face provider
pdm run img-edit \
  --input input.jpg \
  --prompt "Make the cat wear a party hat" \
  --output output.jpg \
  --provider huggingface

# With custom model and parameters
pdm run img-edit \
  --input input.jpg \
  --prompt "Make the cat wear a party hat" \
  --output output.jpg \
  --provider huggingface \
  --model "Qwen/Qwen-Image-Edit" \
  --strength 0.8 \
  --guidance 7.5
```

### Using Explicit Endpoint (Forces Hugging Face)

```bash
# Using explicit endpoint overrides provider selection
pdm run img-edit \
  --input input.jpg \
  --prompt "Make the cat wear a party hat" \
  --output output.jpg \
  --endpoint "https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit"
```

## Provider Discovery

Use the global --providers option:

```bash
# List available providers
pdm run img-edit --providers

# List providers in JSON format
pdm run img-edit --providers --json
```

## Python Code Examples

### Using the Provider Abstraction Layer Directly

```python
import os
from lib.providers import get_provider_class
from lib.image_io import load_image
from PIL import Image

# Get the provider class from registry
provider_class = get_provider_class("replicate")  # or "huggingface"
provider = provider_class()

# Load image
image = load_image("cat.png")

# Edit image using the provider
edited_image = provider.edit_image(
    image=image,
    instructions="Turn the cat into a tiger.",
    strength=0.8,
    guidance_scale=7.5,
    seed=42
)

# Save the result
edited_image.save("tiger_cat.png")
```

### Provider-Specific Configuration

```python
from lib.config import get_settings

settings = get_settings()

# Check available providers
print("Available providers:", list(settings.PROVIDERS.keys()))
print("Default provider:", settings.DEFAULT_PROVIDER)

# Access provider-specific configuration
if "replicate" in settings.PROVIDERS:
    replicate_config = settings.PROVIDERS["replicate"]
    print("Replicate API key configured:", bool(replicate_config.api_key))
    
if "huggingface" in settings.PROVIDERS:
    hf_config = settings.PROVIDERS["huggingface"]
    print("Hugging Face model:", hf_config.model)
```

## Backward Compatibility

The application maintains full backward compatibility with existing Hugging Face configurations:

### Legacy Hugging Face Usage (Still Supported)

```bash
# Traditional Hugging Face usage (still works)
pdm run img-edit \
  --input input.jpg \
  --prompt "Make the cat wear a party hat" \
  --output output.jpg \
  --provider huggingface

# Using legacy environment variables
export HF_TOKEN=your_token_here
pdm run img-edit -i input.jpg -o output.jpg -p "Remove sunglasses"
```

### Environment Variables Compatibility

The application automatically migrates legacy environment variables to the new provider system:

- `HF_TOKEN` or `HF_API_TOKEN` → Hugging Face provider configuration
- `REPLICATE_API_TOKEN` → Replicate provider configuration
- `IMG_EDIT_PROVIDER` → Sets the default provider
- `IMG_EDIT_MODEL` → Sets the default model for Hugging Face provider

## Error Handling Examples

All providers use the same error handling system:

```python
try:
    from lib.edit_pipeline import ImageEditPipeline
    from lib.config import get_settings
    
    pipeline = ImageEditPipeline(get_settings())
    result = pipeline.edit_image(
        input_path="input.jpg",
        output_path="output.jpg",
        prompt="Make the sky blue"
    )
    
    if result["status"] == "success":
        print(f"Success! Edited image saved to {result['output_path']}")
    else:
        print(f"Error: {result['message']}")
        
except Exception as e:
    print(f"Pipeline error: {e}")
    # Error context includes provider information for debugging
```

## Provider Comparison

| Provider | Default Model | Token Variable | Best For |
|----------|---------------|----------------|----------|
| **Replicate** | qwen/qwen-image-edit | `REPLICATE_API_TOKEN` | General use, better pricing |
| **Hugging Face** | Qwen/Qwen-Image-Edit | `HF_TOKEN` | Advanced users, custom endpoints |

## Notes

- The default provider is now **Replicate** for better cost efficiency and performance
- All providers support the same core parameters: strength, guidance, seed, timeout
- Error handling and logging are consistent across all providers
- You can switch providers at any time without changing your application code
- For detailed provider configuration, see [docs/architecture.md](./architecture.md)
- For CLI usage details, see [docs/usage-cli.md](./usage-cli.md)