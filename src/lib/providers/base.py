"""
Abstract base class for image editing providers.

This module defines the ImageEditProvider abstract base class that all
concrete provider implementations must inherit from and implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from PIL import Image


class ImageEditProvider(ABC):
    """
    Abstract base class for image editing providers.

    This class defines the interface that all image editing providers must implement.
    Providers are responsible for taking an input image and text instructions,
    and returning an edited image based on those instructions.

    Attributes:
        name (str): The unique name identifier for this provider.
        description (str): A brief description of the provider's capabilities.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The unique name identifier for this provider.

        Returns:
            str: The provider's unique name (e.g., 'huggingface', 'replicate', etc.)
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        A brief description of the provider's capabilities.

        Returns:
            str: Description of what this provider offers
        """
        pass

    @abstractmethod
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
        Edit an image based on text instructions using the provider's AI model.

        Args:
            image: The input image to be edited (PIL Image object)
            instructions: Text instructions describing the desired edit
            strength: Control strength for the edit (0.0 to 1.0, default: 0.8)
            guidance_scale: Guidance scale for the AI model (default: 7.5)
            seed: Optional random seed for reproducible results
            mask: Optional mask image for selective editing
            **kwargs: Additional provider-specific parameters

        Returns:
            Image.Image: The edited image as a PIL Image object

        Raises:
            ProviderApiError: If the API request fails or returns an error
            NetworkError: If there are network connectivity issues
            ValidationError: If input parameters are invalid
            IoError: If there are issues with image input/output
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up any resources used by the provider.

        This method should be called when the provider is no longer needed
        to release any resources (network connections, file handles, etc.)
        """
        pass