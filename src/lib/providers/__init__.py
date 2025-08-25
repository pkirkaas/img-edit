"""
Providers package for image editing abstraction layer.

This package provides an abstract base class and registry system for
managing multiple image editing provider implementations.

Exports:
    ImageEditProvider: Abstract base class for all image editing providers
    ProviderRegistry: Singleton registry for managing provider classes
"""

from lib.providers.base import ImageEditProvider
from lib.providers.registry import ProviderRegistry

# Import provider implementations to ensure they are registered
# These imports must come after the registry is defined
try:
    from lib.providers.replicate import ReplicateProvider
    from lib.providers.huggingface import HuggingFaceProvider
except ImportError as e:
    print(f"Warning: Failed to import provider implementations: {e}")

# Create a global registry instance for convenience
_registry = ProviderRegistry()

def get_registry() -> ProviderRegistry:
    """
    Get the global provider registry instance.

    Returns:
        ProviderRegistry: The singleton instance of the provider registry
    """
    return _registry

def register_provider(provider_class: type) -> None:
    """
    Register a provider class in the global registry.

    This is a convenience function that uses the global registry instance.

    Args:
        provider_class: The provider class to register, must be a subclass
            of ImageEditProvider

    Raises:
        TypeError: If provider_class is not a subclass of ImageEditProvider
        ValueError: If a provider with the same name is already registered
    """
    _registry.register(provider_class)

def get_provider_class(name: str) -> type:
    """
    Retrieve a provider class by name from the global registry.

    This is a convenience function that uses the global registry instance.

    Args:
        name: The unique name of the provider to retrieve

    Returns:
        type: The provider class implementation

    Raises:
        KeyError: If no provider with the given name is registered
    """
    return _registry.get_provider_class(name)

def list_providers() -> dict:
    """
    List all registered providers with their descriptions.

    This is a convenience function that uses the global registry instance.

    Returns:
        dict: Mapping of provider names to their descriptions
    """
    return _registry.list_providers()

# Define what is exported when using 'from lib.providers import *'
__all__ = [
    'ImageEditProvider',
    'ProviderRegistry',
    'get_registry',
    'register_provider',
    'get_provider_class',
    'list_providers',
]