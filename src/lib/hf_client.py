"""
HTTP client for Hugging Face Serverless Inference API.

This module provides a client for making authenticated HTTP requests to the
Hugging Face Serverless Inference API, specifically for the Qwen/Qwen-Image-Edit model.
It handles authentication, retries, timeouts, and response parsing.

Example usage:
    >>> from lib.hf_client import HFClient
    >>> from lib.config import get_settings
    >>> 
    >>> client = HFClient(get_settings())
    >>> response = client.inference_request(
    ...     image_bytes=b"...",
    ...     prompt="Make the sky blue",
    ...     guidance_scale=7.5,
    ...     strength=0.8
    ... )

Classes:
    HFClient: Main client class for Hugging Face inference requests
    HFClientError: Base exception for client errors
    HFAPIError: Exception for API-related errors
    HFNetworkError: Exception for network-related errors
"""

import base64
import io
import json
import time
from typing import Any, Dict, Optional, Union

import httpx
from PIL import Image

from lib.config import Settings
from lib.logging_utils import get_logger

# Module-level logger
logger = get_logger(__name__)


class HFClientError(Exception):
    """Base exception for Hugging Face client errors."""
    pass


class HFAPIError(HFClientError):
    """Exception for API-related errors from Hugging Face."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        """
        Initialize API error with details.
        
        Args:
            message: Error message
            status_code: HTTP status code if available
            response: Full response data if available
        """
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class HFNetworkError(HFClientError):
    """Exception for network-related errors."""
    pass


class HFClient:
    """
    Client for making requests to Hugging Face Serverless Inference API.
    
    This client handles authentication, request formatting, retries, and error handling
    for the Hugging Face inference API.
    
    Attributes:
        settings: Application settings containing API token and endpoint
        client: HTTPX client instance
        max_retries: Maximum number of retry attempts for failed requests
        retry_delay: Delay between retry attempts in seconds
    """
    
    def __init__(self, settings: Settings, max_retries: int = 3, retry_delay: float = 2.0):
        """
        Initialize the Hugging Face client.
        
        Args:
            settings: Application settings with API configuration
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay between retries in seconds (default: 2.0)
            
        Raises:
            ValueError: If settings are invalid or API token is missing
        """
        if not settings.HF_API_TOKEN:
            raise ValueError("HF_API_TOKEN is required for Hugging Face client")
        
        self.settings = settings
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Create HTTP client with timeout
        self.client = httpx.Client(
            timeout=settings.IMG_EDIT_TIMEOUT_SECONDS,
            headers={
                "Authorization": f"Bearer {settings.HF_API_TOKEN}",
                "Content-Type": "application/json",
            }
        )
        
        logger.debug(f"HFClient initialized with endpoint: {settings.HF_INFERENCE_ENDPOINT}")
    
    def inference_request(
        self,
        image_bytes: bytes,
        prompt: str,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """
        Send an inference request to Hugging Face API for image editing.
        
        Args:
            image_bytes: Raw bytes of the input image
            prompt: Text prompt describing the desired edit
            guidance_scale: Guidance scale parameter (default: from settings)
            strength: Strength parameter for editing (default: from settings)
            seed: Optional seed for deterministic results
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            PIL.Image.Image: The edited image returned by the API
            
        Raises:
            HFAPIError: If the API returns an error response
            HFNetworkError: If there are network issues
            HFClientError: For other client-related errors
            
        Example:
            >>> with open("input.jpg", "rb") as f:
            ...     image_bytes = f.read()
            >>> edited_image = client.inference_request(
            ...     image_bytes=image_bytes,
            ...     prompt="Make the background blue",
            ...     guidance_scale=7.5
            ... )
        """
        # Use defaults from settings if not provided
        guidance_scale = guidance_scale or self.settings.IMG_EDIT_DEFAULT_GUIDANCE
        strength = strength or self.settings.IMG_EDIT_DEFAULT_STRENGTH
        
        # Prepare the request payload
        payload = self._prepare_payload(
            image_bytes=image_bytes,
            prompt=prompt,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
            **kwargs
        )
        
        # Make the request with retry logic
        response_data = self._make_request_with_retry(payload)
        
        # Parse and return the image
        return self._parse_response(response_data)
    
    def _prepare_payload(
        self,
        image_bytes: bytes,
        prompt: str,
        guidance_scale: float,
        strength: float,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare the JSON payload for the inference request.
        
        Args:
            image_bytes: Raw image bytes
            prompt: Text prompt
            guidance_scale: Guidance scale parameter
            strength: Strength parameter
            seed: Optional seed
            **kwargs: Additional parameters
            
        Returns:
            Dict: JSON-serializable payload
        """
        # Encode image as base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        payload = {
            "inputs": {
                "image": image_b64,
                "prompt": prompt,
                "guidance_scale": guidance_scale,
                "strength": strength,
            }
        }
        
        # Add optional parameters if provided
        if seed is not None:
            payload["inputs"]["seed"] = seed
        
        # Add any additional parameters
        payload["inputs"].update(kwargs)
        
        logger.debug(f"Prepared payload with prompt: {prompt[:50]}...")
        return payload
    
    def _make_request_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make the HTTP request with retry logic.
        
        Args:
            payload: Request payload
            
        Returns:
            Dict: Response data from API
            
        Raises:
            HFAPIError: For API errors
            HFNetworkError: For network errors
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(
                    self.settings.HF_INFERENCE_ENDPOINT,
                    json=payload
                )
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse JSON response
                return response.json()
                
            except httpx.HTTPStatusError as e:
                # Handle HTTP errors (4xx, 5xx)
                error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
                logger.error(f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {error_msg}")
                
                if attempt == self.max_retries:
                    raise HFAPIError(
                        f"API request failed after {self.max_retries + 1} attempts: {error_msg}",
                        status_code=e.response.status_code,
                        response=e.response.json() if e.response.content else None
                    )
                
            except (httpx.RequestError, httpx.TimeoutException) as e:
                # Handle network errors
                error_msg = f"Network error: {e}"
                logger.error(f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {error_msg}")
                
                if attempt == self.max_retries:
                    raise HFNetworkError(
                        f"Network error after {self.max_retries + 1} attempts: {error_msg}"
                    )
            
            # Wait before retry
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        # This should never be reached due to the retry logic above
        raise HFClientError("Unexpected error in request retry logic")
    
    def _parse_response(self, response_data: Dict[str, Any]) -> Image.Image:
        """
        Parse the API response and extract the edited image.
        
        Args:
            response_data: Response data from API
            
        Returns:
            PIL.Image.Image: Edited image
            
        Raises:
            HFAPIError: If the response doesn't contain a valid image
        """
        try:
            # The response should contain base64-encoded image data
            if "image" not in response_data:
                raise HFAPIError("API response does not contain image data")
            
            image_b64 = response_data["image"]
            image_bytes = base64.b64decode(image_b64)
            
            # Create PIL Image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            logger.debug(f"Successfully parsed response image: {image.size}")
            return image
            
        except (ValueError, KeyError, Exception) as e:
            raise HFAPIError(f"Failed to parse API response: {e}")
    
    def close(self) -> None:
        """Close the HTTP client connection."""
        self.client.close()
        logger.debug("HFClient connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the client."""
        self.close()


if __name__ == "__main__":
    # Test the HF client (requires valid API token in environment)
    from lib.config import get_settings
    
    try:
        settings = get_settings()
        client = HFClient(settings)
        
        # Create a simple test image
        test_image = Image.new("RGB", (100, 100), color="red")
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format="JPEG")
        image_bytes = img_buffer.getvalue()
        
        print("HFClient initialized successfully")
        print(f"Endpoint: {settings.HF_INFERENCE_ENDPOINT}")
        print(f"Timeout: {settings.IMG_EDIT_TIMEOUT_SECONDS}s")
        
        # Note: Actual API call would require valid token and credits
        # client.inference_request(image_bytes, "test prompt")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        if 'client' in locals():
            client.close()