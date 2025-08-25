"""
Configuration management for the image editing application.

Syntax validated with ast.parse.

This module loads and validates environment-backed settings using Pydantic. It supports
the new InferenceClient approach where configuration can be provided via either:
- provider + token (preferred)
- endpoint (overrides provider/model if present)
- model + token (fallback)

Token resolution prefers HF_TOKEN over HF_API_TOKEN. Includes rich error handling with
context and user guidance for better diagnostics.

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

from typing import Optional, Dict, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from lib.errors import ValidationError, create_error_context

# Load environment variables from .env file (if present)
load_dotenv()


class ConfigurationError(ValidationError):
    """
    Exception type for configuration validation issues.
    
    Raised when configuration settings are invalid, missing, or inconsistent.
    Includes rich context and user guidance for resolution.

    Attributes:
        context: Error context with configuration details
        user_guidance: Helpful guidance for resolving the configuration issue
        
    Example:
        >>> raise ConfigurationError("Missing Hugging Face token", context=ctx, user_guidance="Set HF_TOKEN environment variable")
    """
    pass


class ProviderSettings(BaseSettings):
    """
    Configuration settings for a specific provider.

    Attributes:
        provider_type: The type of provider (e.g., "huggingface", "replicate")
        api_key: API key for the provider (optional for some providers)
        model: Model identifier for the provider (optional)
        endpoint: Custom endpoint URL (optional; overrides provider defaults)
        timeout: Timeout in seconds for API calls (optional; defaults to global timeout)
        extra_config: Additional provider-specific configuration options
    """

    model_config = SettingsConfigDict(
        env_prefix="",  # No prefix for nested settings
        extra="ignore",
        case_sensitive=False,
    )

    provider_type: str = Field(..., description="Provider type identifier")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    model: Optional[str] = Field(default=None, description="Model identifier for the provider")
    endpoint: Optional[str] = Field(default=None, description="Custom endpoint URL for the provider")
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds for API calls")
    extra_config: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific configuration")

    @field_validator("endpoint")
    @classmethod
    def _validate_endpoint(cls, v: Optional[str]) -> Optional[str]:
        """Validate endpoint URL format if provided."""
        if v is None:
            return None
        v2 = v.strip()
        if not v2:
            return None
        if not v2.startswith(("http://", "https://")):
            error_context = create_error_context(
                operation="config_validation",
                request_params={"field": "endpoint", "value": v},
                root_cause=ValueError()
            )
            raise ConfigurationError(
                "Provider endpoint must be a valid HTTP/HTTPS URL when provided",
                context=error_context,
                user_guidance="Provide a valid URL starting with http:// or https://, or leave it unset."
            )
        return v2.rstrip("/")


class Settings(BaseSettings):
    """
    Application settings with environment variable validation.

    Attributes:
        HF_TOKEN: Preferred Hugging Face API token (backward compatibility)
        HF_API_TOKEN: Fallback Hugging Face API token (backward compatibility)
        IMG_EDIT_PROVIDER: Provider identifier (e.g., "replicate") - sets default provider
        IMG_EDIT_MODEL: Model repo id (e.g., "Qwen/Qwen-Image-Edit") - sets default model
        HF_INFERENCE_ENDPOINT: Optional explicit endpoint (overrides provider/model) - backward compatibility
        IMG_EDIT_TIMEOUT_SECONDS: Timeout in seconds for API calls
        IMG_EDIT_DEFAULT_GUIDANCE: Default guidance scale
        IMG_EDIT_DEFAULT_STRENGTH: Default editing strength
        IMG_EDIT_SEED: Optional seed for determinism
        DEFAULT_PROVIDER: The default provider name to use
        PROVIDERS: Dictionary of provider-specific configurations
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
    REPLICATE_API_TOKEN: Optional[str] = Field(default=None, description="Replicate API token for replicate provider")

    # Provider/model approach by default; endpoint optional override
    IMG_EDIT_PROVIDER: str = Field(default="replicate", description="Inference provider name (default: replicate)")
    IMG_EDIT_MODEL: str = Field(default="Qwen/Qwen-Image-Edit", description="Model repo id for image edit")

    HF_INFERENCE_ENDPOINT: Optional[str] = Field(
        default=None,
        description="Optional explicit endpoint URL; if set, overrides provider/model (backward compatibility)",
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

    # New provider configuration system
    DEFAULT_PROVIDER: str = Field(default="replicate", description="Default provider name to use")
    PROVIDERS: Dict[str, ProviderSettings] = Field(
        default_factory=dict,
        description="Dictionary of provider-specific configurations keyed by provider name"
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
            error_context = create_error_context(
                operation="config_validation",
                request_params={"field": "HF_INFERENCE_ENDPOINT", "value": v},
                root_cause=ValueError()
            )
            raise ConfigurationError(
                "HF_INFERENCE_ENDPOINT must be a valid HTTP/HTTPS URL when provided",
                context=error_context,
                user_guidance="Provide a valid URL starting with http:// or https://, or leave it unset to use provider/model."
            )
        return v2.rstrip("/")

    # Validators for provider configuration

    @field_validator("PROVIDERS", mode="after")
    @classmethod
    def _validate_providers(cls, v: Dict[str, ProviderSettings], info) -> Dict[str, ProviderSettings]:
        """Validate provider configurations and ensure default provider exists or gets auto-populated."""
        # Ensure default provider exists in providers dict if specified
        default_provider = info.data.get("DEFAULT_PROVIDER")
        if default_provider and default_provider not in v:
            # Attempt graceful auto-population for known providers to avoid hard failures on simple commands
            if str(default_provider).lower() == "replicate":
                replicate_token = info.data.get("REPLICATE_API_TOKEN")
                if replicate_token:
                    # Auto-populate minimal replicate configuration from environment
                    # Copy dict to avoid mutating the original input mapping unexpectedly
                    v = dict(v)
                    v["replicate"] = ProviderSettings(
                        provider_type="replicate",
                        api_key=replicate_token,
                        model=None,   # Replicate models are referenced differently; leave unset here
                        endpoint=None,
                        timeout=info.data.get("IMG_EDIT_TIMEOUT_SECONDS", 120),
                        extra_config={}
                    )
                else:
                    # No token available; warn instead of raising to let non-edit commands (e.g., "providers") work
                    print("Warning: DEFAULT_PROVIDER is 'replicate' but REPLICATE_API_TOKEN is not set; "
                          "continuing without replicate provider configuration.")
            else:
                # For other providers, warn instead of failing hard so CLI commands like 'providers' still work
                print(f"Warning: Default provider '{default_provider}' is not configured in PROVIDERS. "
                      f"Available: {list(v.keys())}")
        
        # Validate each provider configuration
        for provider_name, provider_settings in v.items():
            if provider_settings.provider_type != provider_name:
                error_context = create_error_context(
                    operation="config_validation",
                    request_params={
                        "field": f"PROVIDERS[{provider_name}].provider_type",
                        "value": provider_settings.provider_type,
                        "expected": provider_name
                    },
                    root_cause=ValueError(f"Provider type mismatch for '{provider_name}'")
                )
                raise ConfigurationError(
                    f"Provider type '{provider_settings.provider_type}' does not match provider name '{provider_name}'",
                    context=error_context,
                    user_guidance="Ensure provider_type matches the dictionary key for each provider"
                )
        
        return v

    @field_validator("DEFAULT_PROVIDER", mode="after")
    @classmethod
    def _validate_default_provider(cls, v: str, info) -> str:
        """Validate that default provider exists in providers dict."""
        providers = info.data.get("PROVIDERS", {})
        if v and v not in providers:
            # Only warn if providers are configured but default is missing
            if providers:
                print(f"Warning: Default provider '{v}' not found in PROVIDERS. Available: {list(providers.keys())}")
        return v

    # Convenience helpers

    def get_token(self) -> Optional[str]:
        """
        Return the effective HF token using HF_TOKEN (preferred) or HF_API_TOKEN fallback.

        Returns:
            Optional[str]: The resolved token value, or None if not set
            
        Raises:
            ConfigurationError: If token validation is required and no token is found
        """
        token = self.HF_TOKEN or self.HF_API_TOKEN
        return token

    def validate_token_required(self) -> None:
        """
        Validate that a Hugging Face token is configured.
        
        Raises:
            ConfigurationError: If no token is configured with guidance
        """
        token = self.get_token()
        if not token:
            error_context = create_error_context(
                operation="config_validation",
                request_params={"field": "HF_TOKEN/HF_API_TOKEN"},
                root_cause=ValueError("No Hugging Face token configured")
            )
            raise ConfigurationError(
                "No Hugging Face token provided",
                context=error_context,
                user_guidance="Set HF_TOKEN (preferred) or HF_API_TOKEN environment variable with your Hugging Face API token."
            )


# Global settings instance (safe even without tokens; token is validated later at client usage)
settings = Settings()

# Backward compatibility: Auto-populate providers from legacy environment variables
def _setup_backward_compatibility():
    """Set up backward compatibility with legacy environment variables."""
    # Create or augment provider configs from legacy environment variables without overriding explicit settings

    # Hugging Face from HF_TOKEN/HF_API_TOKEN if missing
    hf_token = settings.get_token()
    if hf_token and "huggingface" not in settings.PROVIDERS:
        hf_provider = ProviderSettings(
            provider_type="huggingface",
            api_key=hf_token,
            model=settings.IMG_EDIT_MODEL,
            endpoint=settings.HF_INFERENCE_ENDPOINT,
            timeout=settings.IMG_EDIT_TIMEOUT_SECONDS,
            extra_config={}
        )
        settings.PROVIDERS["huggingface"] = hf_provider
    
    # Replicate from REPLICATE_API_TOKEN if missing
    replicate_token = getattr(settings, "REPLICATE_API_TOKEN", None)
    if replicate_token and "replicate" not in settings.PROVIDERS:
        replicate_provider = ProviderSettings(
            provider_type="replicate",
            api_key=replicate_token,
            model=None,  # Replicate uses model IDs differently
            endpoint=None,
            timeout=settings.IMG_EDIT_TIMEOUT_SECONDS,
            extra_config={}
        )
        settings.PROVIDERS["replicate"] = replicate_provider
    
    # Align default provider based on legacy IMG_EDIT_PROVIDER if it maps to a configured provider
    if settings.IMG_EDIT_PROVIDER and settings.IMG_EDIT_PROVIDER in settings.PROVIDERS:
        settings.DEFAULT_PROVIDER = settings.IMG_EDIT_PROVIDER

# Apply backward compatibility setup
_setup_backward_compatibility()


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
    print(f"REPLICATE_API_TOKEN: {('set' if s.REPLICATE_API_TOKEN else 'none')}")
    print(f"Effective HF token resolved: {('set' if s.get_token() else 'none')}")
    print(f"IMG_EDIT_PROVIDER: {s.IMG_EDIT_PROVIDER}")
    print(f"IMG_EDIT_MODEL: {s.IMG_EDIT_MODEL}")
    print(f"HF_INFERENCE_ENDPOINT: {s.HF_INFERENCE_ENDPOINT}")
    print(f"IMG_EDIT_TIMEOUT_SECONDS: {s.IMG_EDIT_TIMEOUT_SECONDS}")
    print(f"IMG_EDIT_DEFAULT_GUIDANCE: {s.IMG_EDIT_DEFAULT_GUIDANCE}")
    print(f"IMG_EDIT_DEFAULT_STRENGTH: {s.IMG_EDIT_DEFAULT_STRENGTH}")
    print(f"IMG_EDIT_SEED: {s.IMG_EDIT_SEED}")
    print(f"DEFAULT_PROVIDER: {s.DEFAULT_PROVIDER}")
    print(f"PROVIDERS: {list(s.PROVIDERS.keys()) if s.PROVIDERS else 'none'}")
    for provider_name, provider_config in s.PROVIDERS.items():
        print(f"  {provider_name}: {provider_config}")