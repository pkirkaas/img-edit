"""
Replicate provider implementation for image editing using Replicate's AI models.

This module provides a concrete implementation of the ImageEditProvider interface
for the Replicate platform, allowing image editing through various AI models
available on Replicate's platform.

Key Features:
- Supports multiple Replicate image editing models
- Configurable via application settings
- Proper error handling and resource management
- Integration with the provider registry system
"""

import io
import logging
from typing import Optional, Dict, Any
from PIL import Image

import replicate
from lib.providers.base import ImageEditProvider
from lib.providers.registry import ProviderRegistry
from lib.config import get_settings, ConfigurationError
from lib.errors import ProviderApiError, NetworkError, ValidationError, IoError, create_error_context

# Set up logging
logger = logging.getLogger(__name__)


@ProviderRegistry.register("replicate")
class ReplicateProvider(ImageEditProvider):
    """
    Image editing provider implementation for Replicate's AI platform.

    This provider uses Replicate's API to perform image editing tasks using
    various AI models. It supports models like "qwen/qwen-image-edit" and
    other image editing models available on Replicate.

    Attributes:
        name (str): The unique provider identifier ("replicate")
        description (str): Description of the provider's capabilities
        _client: The Replicate client instance (initialized on first use)
        _api_token: The Replicate API token from configuration
        _timeout: Request timeout in seconds
    """

    def __init__(self) -> None:
        """
        Initialize a new ReplicateProvider instance.

        The Replicate client is not initialized immediately to allow for
        lazy initialization and proper error handling.

        Raises:
            ConfigurationError: If required configuration is missing
        """
        self._client: Optional[replicate.Client] = None
        self._api_token: Optional[str] = None
        self._timeout: int = 120

        # Load configuration
        settings = get_settings()
        
        # Get Replicate-specific configuration from providers or legacy env vars
        replicate_config = settings.PROVIDERS.get("replicate")
        if replicate_config:
            self._api_token = replicate_config.api_key
            self._timeout = replicate_config.timeout or settings.IMG_EDIT_TIMEOUT_SECONDS
        else:
            # Fallback to legacy environment variable
            self._api_token = settings.REPLICATE_API_TOKEN
            self._timeout = settings.IMG_EDIT_TIMEOUT_SECONDS

        if not self._api_token:
            raise ConfigurationError(
                "Replicate API token is required but not configured",
                user_guidance="Set REPLICATE_API_TOKEN environment variable or configure replicate provider in settings"
            )

    @property
    def name(self) -> str:
        """
        Get the unique name identifier for this provider.

        Returns:
            str: The provider's unique name ("replicate")
        """
        return "replicate"

    @property
    def description(self) -> str:
        """
        Get a brief description of the provider's capabilities.

        Returns:
            str: Description of Replicate provider capabilities
        """
        return "Image editing using Replicate's AI models platform with support for various image editing models"

    def _get_client(self) -> replicate.Client:
        """
        Get or initialize the Replicate client instance.

        Returns:
            replicate.Client: The initialized Replicate client

        Raises:
            ProviderApiError: If client initialization fails
        """
        if self._client is None:
            try:
                self._client = replicate.Client(api_token=self._api_token)
                logger.info("Replicate client initialized successfully")
            except Exception as e:
                error_context = create_error_context(
                    operation="client_initialization",
                    provider=self.name,
                    root_cause=e
                )
                raise ProviderApiError(
                    f"Failed to initialize Replicate client: {str(e)}",
                    context=error_context,
                    user_guidance="Check your Replicate API token and network connection"
                ) from e
        return self._client

    def edit_image(
        self,
        image: Image.Image,
        instructions: str,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        mask: Optional[Image.Image] = None,
        **kwargs: Any
    ) -> Image.Image:
        """
        Edit an image based on text instructions using Replicate's AI models.

        This method uses Replicate's image editing models (default: "qwen/qwen-image-edit")
        to modify an image according to the provided text instructions.

        Args:
            image: The input image to be edited (PIL Image object)
            instructions: Text instructions describing the desired edit
            strength: Control strength for the edit (0.0 to 1.0, default: 0.8)
            guidance_scale: Guidance scale for the AI model (default: 7.5)
            seed: Optional random seed for reproducible results
            mask: Optional mask image for selective editing (not all models support this)
            **kwargs: Additional provider-specific parameters including:
                - model: Specific Replicate model to use (default: "qwen/qwen-image-edit")
                - num_inference_steps: Number of diffusion steps
                - negative_prompt: Negative prompt for guidance

        Returns:
            Image.Image: The edited image as a PIL Image object

        Raises:
            ProviderApiError: If the Replicate API request fails or returns an error
            NetworkError: If there are network connectivity issues
            ValidationError: If input parameters are invalid
            IoError: If there are issues with image input/output processing
        """
        # Validate input parameters
        if not instructions or not instructions.strip():
            raise ValidationError(
                "Instructions cannot be empty",
                user_guidance="Provide meaningful text instructions for the image edit"
            )

        if strength < 0.0 or strength > 1.0:
            raise ValidationError(
                f"Strength must be between 0.0 and 1.0, got {strength}",
                user_guidance="Set strength to a value between 0.0 and 1.0"
            )

        if guidance_scale <= 0:
            raise ValidationError(
                f"Guidance scale must be positive, got {guidance_scale}",
                user_guidance="Set guidance_scale to a positive value"
            )

        # Get model from kwargs or use default
        model_name = kwargs.pop("model", "qwen/qwen-image-edit")

        try:
            client = self._get_client()

            # Prepare input parameters for Replicate
            input_params = {
                "image": self._pil_to_bytes(image),
                "prompt": instructions.strip(),
                "strength": strength,
                "guidance_scale": guidance_scale,
                **kwargs
            }

            # Add optional parameters if provided
            if seed is not None:
                input_params["seed"] = seed

            if mask is not None:
                input_params["mask"] = self._pil_to_bytes(mask)

            logger.info(
                f"Calling Replicate model {model_name} with parameters: "
                f"strength={strength}, guidance_scale={guidance_scale}, "
                f"seed={seed}, instructions_length={len(instructions)}"
            )

            # Execute the prediction
            output = client.run(
                model_name,
                input=input_params,
                timeout=self._timeout
            )

            # Replicate returns a list of outputs, typically containing the edited image URL
            if not output or not isinstance(output, list) or len(output) == 0:
                raise ProviderApiError(
                    "Replicate API returned empty or invalid response",
                    user_guidance="Check the model output format and try again"
                )

            # Download and process the edited image
            edited_image = self._download_image(output[0])
            return edited_image

        except replicate.exceptions.ReplicateError as e:
            error_context = create_error_context(
                operation="image_edit",
                provider=self.name,
                model=model_name,
                request_params={
                    "strength": strength,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "instructions_length": len(instructions)
                },
                root_cause=e
            )
            raise ProviderApiError(
                f"Replicate API error: {str(e)}",
                context=error_context,
                user_guidance="Check your API token, model name, and input parameters"
            ) from e
        except TimeoutError as e:
            error_context = create_error_context(
                operation="image_edit",
                provider=self.name,
                model=model_name,
                request_params={"timeout": self._timeout},
                root_cause=e
            )
            raise NetworkError(
                f"Replicate API request timed out after {self._timeout} seconds",
                context=error_context,
                user_guidance="Increase timeout setting or check network connectivity"
            ) from e
        except Exception as e:
            error_context = create_error_context(
                operation="image_edit",
                provider=self.name,
                model=model_name,
                root_cause=e
            )
            if "connection" in str(e).lower() or "network" in str(e).lower():
                raise NetworkError(
                    f"Network error during Replicate API call: {str(e)}",
                    context=error_context,
                    user_guidance="Check your network connection and try again"
                ) from e
            else:
                raise ProviderApiError(
                    f"Unexpected error during Replicate API call: {str(e)}",
                    context=error_context,
                    user_guidance="Check the error details and try again"
                ) from e

    def _pil_to_bytes(self, image: Image.Image) -> io.BytesIO:
        """
        Convert a PIL Image to bytes for API consumption.

        Args:
            image: PIL Image to convert

        Returns:
            io.BytesIO: Bytes buffer containing the image data

        Raises:
            IoError: If image conversion fails
        """
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            return buffer
        except Exception as e:
            raise IoError(
                f"Failed to convert image to bytes: {str(e)}",
                user_guidance="Check that the image is valid and in a supported format"
            ) from e

    def _download_image(self, image_url: str) -> Image.Image:
        """
        Download an image from a URL and return as PIL Image.

        Args:
            image_url: URL of the image to download

        Returns:
            Image.Image: Downloaded image as PIL Image

        Raises:
            IoError: If image download or processing fails
            ProviderApiError: If the downloaded content is not a valid image
        """
        try:
            import requests
            
            response = requests.get(image_url, timeout=self._timeout)
            response.raise_for_status()
            
            image_data = io.BytesIO(response.content)
            image = Image.open(image_data)
            
            # Convert to RGB if necessary (some models might return RGBA)
            if image.mode in ("RGBA", "LA"):
                image = image.convert("RGB")
            
            return image
            
        except requests.RequestException as e:
            raise IoError(
                f"Failed to download edited image from {image_url}: {str(e)}",
                user_guidance="Check network connectivity and image URL accessibility"
            ) from e
        except Exception as e:
            raise ProviderApiError(
                f"Failed to process downloaded image: {str(e)}",
                user_guidance="The edited image may be corrupted or in an unsupported format"
            ) from e

    def close(self) -> None:
        """
        Clean up any resources used by the provider.

        For Replicate provider, this primarily involves cleaning up the client
        instance if it was initialized. The Replicate client doesn't maintain
        persistent connections, but we still clean up for consistency.
        """
        self._client = None
        logger.info("Replicate provider resources cleaned up")

    def __del__(self) -> None:
        """Ensure resources are cleaned up when the provider is garbage collected."""
        self.close()


# Register the provider with the global registry
# This happens automatically due to the @ProviderRegistry.register decorator