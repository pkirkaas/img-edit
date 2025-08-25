"""
Unified error hierarchy for the image editing application with rich context capture.

This module defines a structured exception hierarchy that captures detailed context
for errors occurring throughout the image editing pipeline. Each exception includes
comprehensive diagnostic information and user-friendly guidance for troubleshooting.

Error Hierarchy:
    BaseError: Root exception for all application errors
    ├── ApiError: Base for API-related errors
    │   ├── ProviderApiError: Errors from specific providers (fal-ai, etc.)
    │   └── HuggingFaceApiError: Errors from Hugging Face API
    ├── PipelineError: Errors in the editing pipeline orchestration
    ├── ValidationError: Input validation errors
    │   ├── ImageValidationError: Image-specific validation issues
    │   ├── ConfigValidationError: Configuration validation issues
    │   └── ParameterValidationError: Parameter validation issues
    ├── NetworkError: Network connectivity issues
    ├── IoError: File I/O operations errors
    └── TimeoutError: Operation timeout errors

Each exception captures:
    - Operation being performed
    - Provider/model/endpoint context
    - HTTP status codes and request IDs when available
    - Execution timing information
    - Sanitized request parameters
    - Response snippets for debugging
    - Root cause exceptions
    - User-friendly remediation guidance

Syntax validated with ast.parse.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List, Union
from datetime import datetime

from lib.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ErrorContext:
    """
    Rich context container for error diagnostics and troubleshooting.
    
    Attributes:
        operation: Name of the operation that failed (e.g., 'image_to_image')
        provider: Inference provider name (e.g., 'fal-ai')
        model: Model identifier (e.g., 'Qwen/Qwen-Image-Edit')
        endpoint: API endpoint URL if explicitly set
        http_status: HTTP status code from API response if available
        request_id: Request identifier from API headers if available
        elapsed_ms: Execution time in milliseconds
        request_params: Sanitized request parameters (sensitive data removed)
        response_snippet: First 500 characters of response body for debugging
        root_cause: Original exception that caused this error
        timestamp: When the error occurred (ISO 8601 format)
    """
    operation: str = "unknown"
    provider: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    http_status: Optional[int] = None
    request_id: Optional[str] = None
    elapsed_ms: Optional[float] = None
    request_params: Dict[str, Any] = field(default_factory=dict)
    response_snippet: Optional[str] = None
    root_cause: Optional[Exception] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        result = asdict(self)
        # Handle root_cause separately since Exception isn't serializable
        if self.root_cause is not None:
            result["root_cause"] = {
                "type": type(self.root_cause).__name__,
                "message": str(self.root_cause)
            }
        return result
    
    def to_json(self) -> str:
        """Serialize context to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class BaseError(Exception):
    """
    Base exception for all application errors with rich context capture.
    
    This exception provides structured error information including diagnostic
    context, user-friendly messages, and remediation guidance.
    
    Attributes:
        message: Primary error message
        context: ErrorContext with detailed diagnostic information
        user_guidance: User-friendly guidance for resolving the issue
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        user_guidance: Optional[str] = None,
    ):
        self.message = message
        self.context = context or ErrorContext()
        self.user_guidance = user_guidance or self._default_guidance()
        
        # Build comprehensive error message
        full_message = self._build_message()
        super().__init__(full_message)
    
    def _default_guidance(self) -> str:
        """Default user guidance for this error type."""
        return "Check the error context for details and review application logs."
    
    def _build_message(self) -> str:
        """Build comprehensive error message with context and guidance."""
        parts = [f"{self.message}"]
        
        # Add context summary if available
        context_summary = self._context_summary()
        if context_summary:
            parts.append(f"Context: {context_summary}")
        
        # Add user guidance
        parts.append(f"Guidance: {self.user_guidance}")
        
        return " | ".join(parts)
    
    def _context_summary(self) -> str:
        """Create a concise summary of the error context."""
        summary_parts = []
        
        if self.context.operation != "unknown":
            summary_parts.append(f"operation={self.context.operation}")
        
        if self.context.provider:
            summary_parts.append(f"provider={self.context.provider}")
        
        if self.context.model:
            summary_parts.append(f"model={self.context.model}")
        
        if self.context.endpoint:
            summary_parts.append(f"endpoint={self.context.endpoint}")
        
        if self.context.http_status:
            summary_parts.append(f"status={self.context.http_status}")
        
        if self.context.request_id:
            summary_parts.append(f"request_id={self.context.request_id}")
        
        if self.context.elapsed_ms:
            summary_parts.append(f"elapsed={self.context.elapsed_ms:.0f}ms")
        
        return ", ".join(summary_parts) if summary_parts else ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for structured logging."""
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "context": self.context.to_dict(),
            "user_guidance": self.user_guidance,
            "timestamp": self.context.timestamp,
        }
    
    def to_json(self) -> str:
        """Serialize error to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ApiError(BaseError):
    """Base class for API-related errors."""
    
    def _default_guidance(self) -> str:
        return (
            "Check your API credentials, quota limits, and network connectivity. "
            "Verify the provider/model is available and your request parameters are valid."
        )


class ProviderApiError(ApiError):
    """Error from specific inference providers (fal-ai, etc.)."""
    
    def _default_guidance(self) -> str:
        guidance = super()._default_guidance()
        if self.context.provider == "fal-ai":
            guidance += (
                " For fal-ai, check your Hugging Face token has access to the model "
                "and you have sufficient credits. Visit https://fal.ai for account status."
            )
        return guidance


class HuggingFaceApiError(ApiError):
    """Error specifically from Hugging Face API."""
    
    def _default_guidance(self) -> str:
        return (
            "Check your Hugging Face token validity and permissions at "
            "https://huggingface.co/settings/tokens. Ensure the model is available "
            "and you're not exceeding rate limits. Response details in context."
        )


class PipelineError(BaseError):
    """Error in the image editing pipeline orchestration."""
    
    def _default_guidance(self) -> str:
        return (
            "Review input parameters and file paths. Check that input images are valid "
            "and output directories are writable. See detailed context for specific failure point."
        )


class ValidationError(BaseError):
    """Base class for validation errors."""
    
    def _default_guidance(self) -> str:
        return "Review the input parameters and ensure they meet the required format and constraints."


class ImageValidationError(ValidationError):
    """Error validating image files."""
    
    def _default_guidance(self) -> str:
        return (
            "Ensure the image file exists, is readable, and in a supported format "
            "(JPEG, PNG, BMP, GIF, TIFF, WebP). Check file permissions and corruption."
        )


class ConfigValidationError(ValidationError):
    """Error validating configuration."""
    
    def _default_guidance(self) -> str:
        return (
            "Check your .env file or environment variables for correct configuration. "
            "Ensure HF_TOKEN or HF_API_TOKEN is set with a valid Hugging Face token."
        )


class ParameterValidationError(ValidationError):
    """Error validating function parameters."""
    
    def _default_guidance(self) -> str:
        return "Review the provided parameters and ensure they meet the expected types and value ranges."


class NetworkError(BaseError):
    """Network connectivity error."""
    
    def _default_guidance(self) -> str:
        return (
            "Check your internet connection and network settings. "
            "Verify the API endpoint is reachable and not blocked by firewall/proxy. "
            "Retry the operation after ensuring network connectivity."
        )


class IoError(BaseError):
    """File I/O operation error."""
    
    def _default_guidance(self) -> str:
        return (
            "Check file permissions and disk space. Ensure source files exist and "
            "destination directories are writable. Verify file paths are correct."
        )


class TimeoutError(BaseError):
    """Operation timeout error."""
    
    def _default_guidance(self) -> str:
        return (
            "The operation took too long to complete. This may be due to network latency, "
            "server load, or large file sizes. Try increasing the timeout setting or "
            "retrying with a smaller image or simpler prompt."
        )


def create_error_context(
    operation: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    http_status: Optional[int] = None,
    request_id: Optional[str] = None,
    start_time: Optional[float] = None,
    request_params: Optional[Dict[str, Any]] = None,
    response_text: Optional[str] = None,
    root_cause: Optional[Exception] = None,
) -> ErrorContext:
    """
    Create an ErrorContext with standardized field population.
    
    Args:
        operation: Name of the operation that failed
        provider: Inference provider name
        model: Model identifier
        endpoint: API endpoint URL
        http_status: HTTP status code from response
        request_id: Request identifier from headers
        start_time: Operation start time (perf_counter) for calculating elapsed time
        request_params: Request parameters (will be sanitized)
        response_text: Full response text for snippet extraction
        root_cause: Original exception that caused the error
    
    Returns:
        ErrorContext: Populated with provided values and calculated fields
    """
    # Sanitize request parameters (remove sensitive data)
    sanitized_params = _sanitize_params(request_params or {})
    
    # Calculate elapsed time if start time provided
    elapsed_ms = None
    if start_time is not None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Extract response snippet
    response_snippet = None
    if response_text:
        response_snippet = response_text[:500] + ("..." if len(response_text) > 500 else "")
    
    return ErrorContext(
        operation=operation,
        provider=provider,
        model=model,
        endpoint=endpoint,
        http_status=http_status,
        request_id=request_id,
        elapsed_ms=elapsed_ms,
        request_params=sanitized_params,
        response_snippet=response_snippet,
        root_cause=root_cause,
    )


def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize request parameters by removing sensitive information.
    
    Args:
        params: Original request parameters
    
    Returns:
        Dict[str, Any]: Sanitized parameters with sensitive fields masked
    """
    sanitized = params.copy()
    
    # Mask sensitive fields
    sensitive_keys = ["token", "api_key", "password", "secret", "key"]
    for key in list(sanitized.keys()):
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = "***MASKED***"
    
    # Truncate large values
    for key, value in sanitized.items():
        if isinstance(value, str) and len(value) > 100:
            sanitized[key] = value[:100] + "..."
        elif isinstance(value, (list, dict)) and len(str(value)) > 100:
            sanitized[key] = f"{type(value).__name__}[{len(value)}]"
    
    return sanitized


def save_error_artifact(error: BaseError, filename: str) -> None:
    """
    Save error details to a file for later analysis.
    
    Args:
        error: The error to save
        filename: Path to save the error artifact
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(error.to_json())
        logger.info(f"Error artifact saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save error artifact to {filename}: {e}")


def format_error_for_cli(error: Exception, verbose: bool = False) -> str:
    """
    Format error for CLI display with appropriate detail level.
    
    Args:
        error: The error to format (can be BaseError or a standard Exception)
        verbose: Whether to include detailed context
    
    Returns:
        str: Formatted error message for CLI output
    """
    # Handle structured application errors with rich context
    if isinstance(error, BaseError):
        if verbose:
            # Full detailed output with serialized context
            return f"{type(error).__name__}: {error}\n\nFull context:\n{error.to_json()}"
        else:
            # Concise user-friendly output
            return f"{type(error).__name__}: {error.message} | {error.user_guidance}"
    
    # Fallback for regular Exceptions without custom attributes
    if verbose:
        tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        return f"{type(error).__name__}: {error}\n\nTraceback:\n{tb}"
    else:
        return f"{type(error).__name__}: {str(error)}"