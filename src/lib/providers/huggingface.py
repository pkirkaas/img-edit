"""
Hugging Face provider implementation for image editing using Hugging Face Inference API.

This module provides a concrete implementation of the ImageEditProvider interface
for the Hugging Face platform, wrapping the existing HFImageEditClient for backward
compatibility while adapting it to the new provider abstraction layer.

Key Features:
- Wraps HFImageEditClient for seamless transition to provider system
- Supports Hugging Face Inference API with provider/model configuration
- Proper error handling and resource management
- Integration with the provider registry system
"""

import io
import logging
from typing import Optional, Dict, Any

from PIL import Image
from lib.providers.base import ImageEditProvider
from lib.providers.registry import ProviderRegistry
from lib.config import get_settings, ConfigurationError
from lib.errors import ProviderApiError, NetworkError, ValidationError, IoError
from lib.hf_client import HFImageEditClient

# Set up logging
logger = logging.getLogger(__name__)


@ProviderRegistry.register("huggingface")
class HuggingFaceProvider(ImageEditProvider):
    """
    Image editing provider implementation for Hugging Face's Inference API.

    This provider wraps the existing HFImageEditClient to provide compatibility
    with the new provider abstraction system. It supports Hugging Face's
    image editing models through the Inference API.

    Attributes:
        name (str): The unique provider identifier ("huggingface")
        description (str): Description of the provider's capabilities
        _client: The HFImageEditClient instance (initialized on first use)
        _api_token: The Hugging Face API token from configuration
        _timeout: Request timeout in seconds
        _provider_config: Provider-specific configuration from settings
    """

    def __init__(self) -> None:
        """
        Initialize a new HuggingFaceProvider instance.

        The HFImageEditClient is not initialized immediately to allow for
        lazy initialization and proper error handling.

        Raises:
            ConfigurationError: If required configuration is missing
        """
        self._client: Optional[HFImageEditClient] = None
        self._api_token: Optional[str] = None
        self._timeout: int = 120
        self._provider_config: Optional[Dict[str, Any]] = None

        # Load configuration
        settings = get_settings()
        
        # Get Hugging Face-specific configuration from providers or legacy env vars
        hf_config = settings.PROVIDERS.get("huggingface")
        if hf_config:
            self._api_token = hf_config.api_key
            self._timeout = hf_config.timeout or settings.IMG_EDIT_TIMEOUT_SECONDS
            self._provider_config = {
                "provider": hf_config.provider_type,
                "model": hf_config.model,
                "endpoint": hf_config.endpoint
            }
        else:
            # Fallback to legacy environment variables for backward compatibility
            self._api_token = settings.get_token()
            self._timeout = settings.IMG_EDIT_TIMEOUT_SECONDS
            self._provider_config = {
                "provider": settings.IMG_EDIT_PROVIDER,
                "model": settings.IMG_EDIT_MODEL,
                "endpoint": settings.HF_INFERENCE_ENDPOINT
            }

        if not self._api_token:
            raise ConfigurationError(
                "Hugging Face API token is required but not configured",
                user_guidance="Set HF_TOKEN or HF_API_TOKEN environment variable or configure huggingface provider in settings"
            )

    @property
    def name(self) -> str:
        """
        Get the unique name identifier for this provider.

        Returns:
            str: The provider's unique name ("huggingface")
        """
        return "huggingface"

    @property
    def description(self) -> str:
        """
        Get a brief description of the provider's capabilities.

        Returns:
            str: Description of Hugging Face provider capabilities
        """
        return "Image editing using Hugging Face's Inference API with support for various image editing models like Qwen/Qwen-Image-Edit"

    def _get_client(self) -> HFImageEditClient:
        """
        Get or initialize the HFImageEditClient instance.

        Returns:
            HFImageEditClient: The initialized HFImageEditClient

        Raises:
            ProviderApiError: If client initialization fails
        """
        if self._client is None:
            try:
                self._client = HFImageEditClient(
                    provider=self._provider_config.get("provider"),
                    model=self._provider_config.get("model"),
                    endpoint=self._provider_config.get("endpoint"),
                    token=self._api_token,
                    timeout=self._timeout
                )
                logger.info("HFImageEditClient initialized successfully for HuggingFaceProvider")
            except Exception as e:
                error_context = {
                    "operation": "client_initialization",
                    "provider": self.name,
                    "error_type": type(e).__name__
                }
                raise ProviderApiError(
                    f"Failed to initialize HFImageEditClient: {str(e)}",
                    context=error_context,
                    user_guidance="Check your Hugging Face API token and configuration"
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
        Edit an image based on text instructions using Hugging Face's Inference API.

        This method uses Hugging Face's image editing models (default: "Qwen/Qwen-Image-Edit")
        to modify an image according to the provided text instructions.

        Args:
            image: The input image to be edited (PIL Image object)
            instructions: Text instructions describing the desired edit
            strength: Control strength for the edit (0.0 to 1.0, default: 0.8)
            guidance_scale: Guidance scale for the AI model (default: 7.5)
            seed: Optional random seed for reproducible results
            mask: Optional mask image for selective editing (not all models support this)
            **kwargs: Additional provider-specific parameters including:
                - model: Specific model to use (overrides default)
                - provider: Specific provider to use (overrides default)
                - endpoint: Specific endpoint to use (overrides default)

        Returns:
            Image.Image: The edited image as a PIL Image object

        Raises:
            ProviderApiError: If the Hugging Face API request fails or returns an error
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

        try:
            client = self._get_client()

            # Convert PIL Image to bytes for HF client compatibility
            image_bytes = self._pil_to_bytes(image)

            # Prepare parameters for HF client
            hf_kwargs = kwargs.copy()
            if mask is not None:
                hf_kwargs["mask"] = self._pil_to_bytes(mask)

            logger.info(
                f"Calling Hugging Face API with parameters: "
                f"strength={strength}, guidance_scale={guidance_scale}, "
                f"seed={seed}, instructions_length={len(instructions)}"
            )

            # Execute the image edit using the HF client
            edited_image = client.edit_image(
                input_image=image_bytes,
                prompt=instructions,
                strength=strength,
                guidance=guidance_scale,
                seed=seed,
                **hf_kwargs
            )

            return edited_image

        except (ProviderApiError, NetworkError):
            # Re-raise these as they are already properly formatted
            raise
        except Exception as e:
            error_context = {
                "operation": "image_edit",
                "provider": self.name,
                "parameters": {
                    "strength": strength,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "instructions_length": len(instructions)
                },
                "error_type": type(e).__name__
            }
            if "connection" in str(e).lower() or "network" in str(e).lower():
                raise NetworkError(
                    f"Network error during Hugging Face API call: {str(e)}",
                    context=error_context,
                    user_guidance="Check your network connection and try again"
                ) from e
            else:
                raise ProviderApiError(
                    f"Unexpected error during Hugging Face API call: {str(e)}",
                    context=error_context,
                    user_guidance="Check the error details and try again"
                ) from e

    def _pil_to_bytes(self, image: Image.Image) -> bytes:
        """
        Convert a PIL Image to bytes for API consumption.

        Args:
            image: PIL Image to convert

        Returns:
            bytes: Raw image bytes

        Raises:
            IoError: If image conversion fails
        """
        try:
            buffer = io.BytesIO()
            # Use PNG format to preserve quality and support transparency
            image.save(buffer, format="PNG")
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            raise IoError(
                f"Failed to convert image to bytes: {str(e)}",
                user_guidance="Check that the image is valid and in a supported format"
            ) from e

    def close(self) -> None:
        """
        Clean up any resources used by the provider.

        For Hugging Face provider, this involves closing the HFImageEditClient
        if it was initialized.
        """
        if self._client is not None:
            self._client.close()
            self._client = None
        logger.info("Hugging Face provider resources cleaned up")

    def __del__(self) -> None:
        """Ensure resources are cleaned up when the provider is garbage collected."""
        self.close()


# Register the provider with the global registry
# This happens automatically due to the @ProviderRegistry.register decorator