# Example Python code from Hugging Face for API call to Qwen-Image-Edit

Below is the example code/template from Hugging Face to use the qwen image edit LLM with the hugging face API

Note the Hugging Face API Token in this actual project is located in `.env`, with the key: `HF_API_KEY`

## Example Code
```py
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="fal-ai",
    api_key=os.environ["HF_TOKEN"],
)

with open("cat.png", "rb") as image_file:
   input_image = image_file.read()

# output is a PIL.Image object
image = client.image_to_image(
    input_image,
    prompt="Turn the cat into a tiger.",
    model="Qwen/Qwen-Image-Edit",
)
```