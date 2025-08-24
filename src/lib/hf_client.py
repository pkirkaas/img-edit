"""
Thin wrapper around huggingface_hub.InferenceClient for image editing.

Syntax validated with ast.parse.

This module replaces prior custom HTTP logic with a minimal client that delegates
to huggingface_hub.InferenceClient. It supports provider/model/endpoint modes and
exposes a single high-level method for image editing suitable for Qwen/Qwen-Image-Edit.

Classes:
    HFImageEditClient: Thin wrapper around huggingface_hub.InferenceClient

Uses the unified error hierarchy from lib.errors for rich context capture.
"""

from __future__ import annotations

import os
import time
import json
from typing import Optional, Dict, Any

from PIL import Image
from huggingface_hub import InferenceClient

from lib.logging_utils import get_logger
from lib.errors import (
    ProviderApiError, NetworkError, ValidationError,
    create_error_context, save_error_artifact
)

# Module logger
logger = get_logger(__name__)


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
      - Uses rich error context capture with detailed diagnostics for troubleshooting

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

        start_time = time.perf_counter()
        
        try:
            # InferenceClient.image_to_image returns a PIL.Image.Image
            img = self._client.image_to_image(input_image, **options)
            if not isinstance(img, Image.Image):
                error_context = create_error_context(
                    operation="image_to_image",
                    provider=self._provider,
                    model=eff_model,
                    endpoint=self._endpoint,
                    start_time=start_time,
                    request_params=options,
                    response_text=f"Provider returned non-image result: {type(img)}"
                )
                raise ProviderApiError(
                    "Provider returned a non-image result",
                    context=error_context,
                    user_guidance="This may indicate a provider API change or configuration issue. Check model compatibility."
                )
            return img
            
        except (TimeoutError, OSError, ConnectionError) as e:
            # Network-related errors
            error_context = create_error_context(
                operation="image_to_image",
                provider=self._provider,
                model=eff_model,
                endpoint=self._endpoint,
                start_time=start_time,
                request_params=options,
                root_cause=e
            )
            raise NetworkError(
                f"Network error during image editing: {e}",
                context=error_context,
                user_guidance="Check your internet connection and network settings. The API endpoint may be temporarily unavailable."
            ) from e
            
        except Exception as e:
            # API-level errors - attempt to extract detailed information
            error_message = str(e)
            http_status = None
            request_id = None
            response_text = error_message
            
            # Try to extract HTTP status from common error formats
            if hasattr(e, 'response'):
                try:
                    if hasattr(e.response, 'status_code'):
                        http_status = e.response.status_code
                    if hasattr(e.response, 'headers'):
                        request_id = e.response.headers.get('x-request-id')
                    if hasattr(e.response, 'text'):
                        response_text = e.response.text
                except:
                    pass
            
            # Try to parse JSON error responses for more details
            parsed_error = None
            try:
                if response_text and response_text.strip().startswith('{'):
                    parsed_error = json.loads(response_text)
                    if 'error' in parsed_error:
                        error_message = parsed_error['error']
                    elif 'message' in parsed_error:
                        error_message = parsed_error['message']
            except:
                pass
            
            error_context = create_error_context(
                operation="image_to_image",
                provider=self._provider,
                model=eff_model,
                endpoint=self._endpoint,
                http_status=http_status,
                request_id=request_id,
                start_time=start_time,
                request_params=options,
                response_text=response_text,
                root_cause=e
            )
            
            # Save error artifact for debugging
            try:
                save_error_artifact(
                    ProviderApiError("API error during image editing", context=error_context),
                    "tmp/last_api_error.json"
                )
            except Exception as save_error:
                logger.warning(f"Failed to save error artifact: {save_error}")
            
            raise ProviderApiError(
                f"API error during image editing: {error_message}",
                context=error_context,
                user_guidance=(
                    "This may indicate invalid parameters, authentication issues, or provider limitations. "
                    "Check your API token, parameter values, and ensure the model supports your request. "
                    f"Detailed error context saved to tmp/last_api_error.json"
                )
            ) from e

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