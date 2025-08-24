"""
Configuration management for the image editing application.

Syntax validated with ast.parse.

This module loads and validates environment-backed settings using Pydantic. It supports
the new InferenceClient approach where configuration can be provided via either:
- provider + token (preferred)
- endpoint (overrides provider/model if present)
- model + token (fallback)

Token resolution prefers HF_TOKEN over HF_API_TOKEN.

Environment variables (preferred names):
- HF_TOKEN (preferred) / HF_API_TOKEN (fallback)
- IMG_EDIT_PROVIDER (default: "fal-ai")
- IMG_EDIT_MODEL (default: "Qwen/Qwen-Image-Edit")
- HF_INFERENCE_ENDPOINT (optional; if set, overrides provider/model)
- IMG_EDIT_TIMEOUT_SECONDS (default: 120)
- IMG_EDIT_DEFAULT_GUIDANCE (default: 7.5)
- IMG_EDIT_DEFAULT_STRENGTH (default: 0.8)
- IMG_EDIT_SEED (optional)
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings with environment variable validation.

    Attributes:
        HF_TOKEN: Preferred Hugging Face API token
        HF_API_TOKEN: Fallback Hugging Face API token
        IMG_EDIT_PROVIDER: Provider identifier (e.g., "fal-ai")
        IMG_EDIT_MODEL: Model repo id (e.g., "Qwen/Qwen-Image-Edit")
        HF_INFERENCE_ENDPOINT: Optional explicit endpoint (overrides provider/model)
        IMG_EDIT_TIMEOUT_SECONDS: Timeout in seconds for API calls
        IMG_EDIT_DEFAULT_GUIDANCE: Default guidance scale
        IMG_EDIT_DEFAULT_STRENGTH: Default editing strength
        IMG_EDIT_SEED: Optional seed for determinism
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Token handling: prefer HF_TOKEN, fallback HF_API_TOKEN
    HF_TOKEN: Optional[str] = Field(default=None, description="Preferred Hugging Face API token (recommended name)")
    HF_API_TOKEN: Optional[str] = Field(default=None, description="Fallback token name for Hugging Face API")

    # Provider/model approach by default; endpoint optional override
    IMG_EDIT_PROVIDER: str = Field(default="fal-ai", description="Inference provider name (default: fal-ai)")
    IMG_EDIT_MODEL: str = Field(default="Qwen/Qwen-Image-Edit", description="Model repo id for image edit")

    HF_INFERENCE_ENDPOINT: Optional[str] = Field(
        default=None,
        description="Optional explicit endpoint URL; if set, overrides provider/model",
    )

    IMG_EDIT_TIMEOUT_SECONDS: int = Field(
        default=120,
        description="Timeout in seconds for API calls",
        gt=0,
    )

    IMG_EDIT_DEFAULT_GUIDANCE: float = Field(
        default=7.5,
        description="Default guidance scale for image editing",
        gt=0,
    )

    IMG_EDIT_DEFAULT_STRENGTH: float = Field(
        default=0.8,
        description="Default strength parameter for image editing (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

    IMG_EDIT_SEED: Optional[int] = Field(
        default=None,
        description="Optional seed for deterministic image generation results",
    )

    # Validators

    @field_validator("HF_TOKEN", "HF_API_TOKEN")
    @classmethod
    def _strip_token(cls, v: Optional[str]) -> Optional[str]:
        """Normalize token fields; empty strings become None."""
        if v is None:
            return None
        v2 = v.strip()
        return v2 or None

    @field_validator("HF_INFERENCE_ENDPOINT")
    @classmethod
    def _validate_endpoint(cls, v: Optional[str]) -> Optional[str]:
        """Validate endpoint URL format if provided."""
        if v is None:
            return None
        v2 = v.strip()
        if not v2:
            return None
        if not v2.startswith(("http://", "https://")):
            raise ValueError("HF_INFERENCE_ENDPOINT must be a valid HTTP/HTTPS URL when provided")
        return v2.rstrip("/")

    # Convenience helpers

    def get_token(self) -> Optional[str]:
        """
        Return the effective HF token using HF_TOKEN (preferred) or HF_API_TOKEN fallback.

        Returns:
            Optional[str]: The resolved token value, or None if not set
        """
        return self.HF_TOKEN or self.HF_API_TOKEN


# Global settings instance (safe even without tokens; token is validated later at client usage)
settings = Settings()


def get_settings() -> Settings:
    """
    Get the global application settings instance.

    Returns:
        Settings: The global settings instance
    """
    return settings


if __name__ == "__main__":
    # Test the configuration loading
    s = Settings()
    print("Configuration loaded successfully:")
    print(f"HF_TOKEN (preferred): {('set' if s.HF_TOKEN else 'none')}")
    print(f"HF_API_TOKEN (fallback): {('set' if s.HF_API_TOKEN else 'none')}")
    print(f"Effective token resolved: {('set' if s.get_token() else 'none')}")
    print(f"IMG_EDIT_PROVIDER: {s.IMG_EDIT_PROVIDER}")
    print(f"IMG_EDIT_MODEL: {s.IMG_EDIT_MODEL}")
    print(f"HF_INFERENCE_ENDPOINT: {s.HF_INFERENCE_ENDPOINT}")
    print(f"IMG_EDIT_TIMEOUT_SECONDS: {s.IMG_EDIT_TIMEOUT_SECONDS}")
    print(f"IMG_EDIT_DEFAULT_GUIDANCE: {s.IMG_EDIT_DEFAULT_GUIDANCE}")
    print(f"IMG_EDIT_DEFAULT_STRENGTH: {s.IMG_EDIT_DEFAULT_STRENGTH}")
    print(f"IMG_EDIT_SEED: {s.IMG_EDIT_SEED}")