"""
Configuration management for the image editing application.

This module handles loading and validation of environment variables and application settings
using Pydantic for robust configuration management.

Example usage:
    >>> from lib.config import Settings
    >>> settings = Settings()
    >>> print(settings.HF_API_TOKEN)
    'your_hugging_face_token'

Environment variables:
    HF_API_TOKEN: Required Hugging Face API token for serverless inference
    HF_INFERENCE_ENDPOINT: Hugging Face inference endpoint (default: Qwen/Qwen-Image-Edit)
    IMG_EDIT_TIMEOUT_SECONDS: Request timeout in seconds (default: 120)
    IMG_EDIT_DEFAULT_GUIDANCE: Default guidance scale for image generation (default: 7.5)
    IMG_EDIT_DEFAULT_STRENGTH: Default strength for image editing (default: 0.8)
    IMG_EDIT_SEED: Optional seed for deterministic results
"""

import os
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings with environment variable validation.
    
    This class validates and provides access to all configuration settings
    required for the image editing application.
    
    Attributes:
        HF_API_TOKEN: Hugging Face API token for authentication (required)
        HF_INFERENCE_ENDPOINT: Endpoint for Hugging Face serverless inference
        IMG_EDIT_TIMEOUT_SECONDS: Timeout for HTTP requests in seconds
        IMG_EDIT_DEFAULT_GUIDANCE: Default guidance scale for image generation
        IMG_EDIT_DEFAULT_STRENGTH: Default strength parameter for image editing
        IMG_EDIT_SEED: Optional seed for deterministic generation
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
    
    HF_API_TOKEN: str = Field(
        ...,
        description="Hugging Face API token for serverless inference authentication",
        min_length=1,
    )
    
    HF_INFERENCE_ENDPOINT: str = Field(
        default="https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit",
        description="Hugging Face serverless inference endpoint URL",
    )
    
    IMG_EDIT_TIMEOUT_SECONDS: int = Field(
        default=120,
        description="Timeout in seconds for HTTP requests to Hugging Face API",
        gt=0,
    )
    
    IMG_EDIT_DEFAULT_GUIDANCE: float = Field(
        default=7.5,
        description="Default guidance scale for image generation (controls creativity vs. adherence to prompt)",
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
    
    @field_validator("HF_API_TOKEN")
    @classmethod
    def validate_api_token(cls, v: str) -> str:
        """
        Validate that the API token is provided and not empty.
        
        Args:
            v: The API token value to validate
            
        Returns:
            The validated API token
            
        Raises:
            ValueError: If the API token is empty or missing
        """
        if not v or v.strip() == "":
            raise ValueError("HF_API_TOKEN is required and cannot be empty")
        return v.strip()
    
    @field_validator("HF_INFERENCE_ENDPOINT")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """
        Validate that the endpoint URL is properly formatted.
        
        Args:
            v: The endpoint URL to validate
            
        Returns:
            The validated endpoint URL
            
        Raises:
            ValueError: If the endpoint URL is invalid
        """
        if not v.startswith(("http://", "https://")):
            raise ValueError("HF_INFERENCE_ENDPOINT must be a valid HTTP/HTTPS URL")
        return v.rstrip("/")


# Global settings instance
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
    test_settings = Settings()
    print("Configuration loaded successfully:")
    print(f"HF_API_TOKEN: {test_settings.HF_API_TOKEN[:10]}... (truncated)")
    print(f"HF_INFERENCE_ENDPOINT: {test_settings.HF_INFERENCE_ENDPOINT}")
    print(f"IMG_EDIT_TIMEOUT_SECONDS: {test_settings.IMG_EDIT_TIMEOUT_SECONDS}")
    print(f"IMG_EDIT_DEFAULT_GUIDANCE: {test_settings.IMG_EDIT_DEFAULT_GUIDANCE}")
    print(f"IMG_EDIT_DEFAULT_STRENGTH: {test_settings.IMG_EDIT_DEFAULT_STRENGTH}")
    print(f"IMG_EDIT_SEED: {test_settings.IMG_EDIT_SEED}")