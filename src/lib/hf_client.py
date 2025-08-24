"""
Thin wrapper around huggingface_hub.InferenceClient for image editing.

Syntax validated with ast.parse.

This module replaces prior custom HTTP logic with a minimal client that delegates
to huggingface_hub.InferenceClient. It supports provider/model/endpoint modes and
exposes a single high-level method for image editing suitable for Qwen/Qwen-Image-Edit.

Classes:
    HFClientError: Base exception for client errors
    HFAPIError: Exception for API-related errors
    HFNetworkError: Exception for network-related errors
    HFImageEditClient: Thin wrapper around huggingface_hub.InferenceClient
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any

from PIL import Image
from huggingface_hub import InferenceClient

from lib.logging_utils import get_logger

# Module logger
logger = get_logger(__name__)


class HFClientError(Exception):
    """
    Base exception for Hugging Face client errors.

    Raised when a client operation fails due to configuration, invalid arguments,
    or unexpected runtime issues outside of pure API/network failures.
    """
    pass


class HFAPIError(HFClientError):
    """
    Exception raised for API-related errors (server responded but failed the request).

    Attributes:
        context: Additional context describing the active client configuration (provider/endpoint/model)
        status_code: Optional HTTP-like status code if discernible
        response: Optional structured payload with error details
    """
    def __init__(self, message: str, context: Optional[str] = None, status_code: Optional[int] = None, response: Optional[Dict[str, Any]] = None) -> None:
        self.context = context
        self.status_code = status_code
        self.response = response
        super().__init__(f"{message}" + (f" | context={context}" if context else ""))


class HFNetworkError(HFClientError):
    """
    Exception raised for network-related errors (timeouts, connection errors).
    """
    pass


class HFImageEditClient:
    """
    Thin wrapper around huggingface_hub.InferenceClient for image editing (image-to-image).

    This client centralizes construction of InferenceClient using one of three modes:
      1) endpoint + token (overrides everything)
      2) provider + token (preferred)
      3) model + token (fallback, no provider)

    Token loading:
      - If token is not provided explicitly, reads os.environ["HF_TOKEN"], fallback to os.environ.get("HF_API_TOKEN")

    Methods:
      - edit_image(...): Perform image-to-image editing and return a PIL.Image.Image

    Notes:
      - Parameters are mapped to Qwen/Qwen-Image-Edit expected names:
          strength -> strength
          guidance -> guidance_scale
          seed     -> seed
      - Mask is not used at the moment unless the underlying model supports it; any extra kwargs
        provided by callers will be forwarded to InferenceClient.image_to_image for future compatibility.

    Example:
      >>> client = HFImageEditClient(provider="fal-ai", model="Qwen/Qwen-Image-Edit", endpoint=None, token=None, timeout=120)
      >>> img = client.edit_image(input_image=b"...", prompt="Make it sunset", strength=0.8, guidance=7.5, seed=42)
      >>> assert isinstance(img, Image.Image)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Initialize the HFImageEditClient.

        Args:
            provider: Inference provider name (e.g., "fal-ai"). Used with api_key auth.
            model: Model repo id (e.g., "Qwen/Qwen-Image-Edit"). Used if no provider/endpoint.
            endpoint: Explicit endpoint URL. If provided, takes precedence over provider/model.
            token: Authentication token. If None, resolves from env HF_TOKEN or HF_API_TOKEN.
            timeout: Optional request timeout in seconds; forwarded to InferenceClient.

        Raises:
            ValueError: If no token could be resolved when required to construct the client.
        """
        # Resolve token preference HF_TOKEN -> HF_API_TOKEN
        resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
        if not resolved_token:
            raise ValueError("No Hugging Face token provided. Set HF_TOKEN or HF_API_TOKEN in the environment.")

        self._provider = provider
        self._endpoint = endpoint
        self._model = model or "Qwen/Qwen-Image-Edit"
        self._token = resolved_token
        self._timeout = timeout

        # Instantiate the underlying InferenceClient based on priority: endpoint > provider > model
        self._client: InferenceClient
        if self._endpoint:
            # Support both 'endpoint' and older 'base_url' constructor keyword as a compatibility fallback.
            try:
                self._client = InferenceClient(endpoint=self._endpoint, token=self._token, timeout=self._timeout)
            except TypeError:
                # Older clients may use base_url
                self._client = InferenceClient(base_url=self._endpoint, token=self._token, timeout=self._timeout)
            logger.debug(f"Initialized InferenceClient via endpoint: {self._endpoint}")
        elif self._provider:
            # Provider style: uses api_key for some providers (e.g., fal-ai) in HF 0.24+
            try:
                self._client = InferenceClient(provider=self._provider, api_key=self._token, timeout=self._timeout)
            except TypeError:
                # Fallback if api_key isn't supported; use token param
                self._client = InferenceClient(provider=self._provider, token=self._token, timeout=self._timeout)
            logger.debug(f"Initialized InferenceClient via provider: {self._provider}")
        else:
            # Plain model usage
            self._client = InferenceClient(model=self._model, token=self._token, timeout=self._timeout)
            logger.debug(f"Initialized InferenceClient via model: {self._model}")

    def edit_image(
        self,
        input_image: bytes,
        prompt: str,
        strength: Optional[float] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Edit an image using image-to-image with Qwen/Qwen-Image-Edit via InferenceClient.

        Args:
            input_image: Raw input image bytes
            prompt: Text prompt describing the desired edit
            strength: Optional strength parameter for editing
            guidance: Optional guidance scale (mapped to guidance_scale)
            seed: Optional seed for determinism
            model: Optional model override; defaults to the client model or "Qwen/Qwen-Image-Edit"
            **kwargs: Extra provider/model-specific parameters forwarded to image_to_image

        Returns:
            PIL.Image.Image: The edited image

        Raises:
            HFAPIError: If the provider/model responds with an API-level failure
            HFNetworkError: For network-type errors (timeouts, connection issues)
            HFClientError: For unexpected local client errors
        """
        eff_model = model or self._model or "Qwen/Qwen-Image-Edit"
        ctx = self._build_context()

        # Build options, only set keys when a value is provided to avoid overriding provider defaults
        options: Dict[str, Any] = {"prompt": prompt, "model": eff_model}
        if strength is not None:
            options["strength"] = strength
        if guidance is not None:
            options["guidance_scale"] = guidance
        if seed is not None:
            options["seed"] = seed
        # Forward any other kwargs (e.g., potential future 'mask') without validation for extension
        options.update(kwargs)

        try:
            # InferenceClient.image_to_image returns a PIL.Image.Image
            img = self._client.image_to_image(input_image, **options)
            if not isinstance(img, Image.Image):
                raise HFAPIError("Provider returned a non-image result", context=ctx)
            return img
        except (TimeoutError, OSError) as e:
            # Heuristic classification of network-like errors
            raise HFNetworkError(f"Network error during image_to_image: {e}") from e
        except Exception as e:
            # Treat any other exception as API-level unless clearly network
            msg = f"API error during image_to_image: {e}"
            raise HFAPIError(msg, context=ctx) from e

    def _build_context(self) -> str:
        """Return a concise context string about how the client was configured."""
        if self._endpoint:
            return f"endpoint={self._endpoint}"
        if self._provider:
            return f"provider={self._provider}, model={self._model}"
        return f"model={self._model}"

    # Compatibility surface for with-statement and pipeline cleanup
    def close(self) -> None:
        """
        No-op close for compatibility with prior client interface.

        InferenceClient does not require explicit close; we keep this method so callers
        can safely call client.close().
        """
        return None

    def __enter__(self) -> "HFImageEditClient":
        """Context manager entry (returns self)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit (no-op)."""
        self.close()