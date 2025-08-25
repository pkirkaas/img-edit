"""
Provider registry system for managing image editing provider implementations.

This module implements a registry pattern to dynamically register and lookup
image editing provider classes by their unique names.
"""

from typing import Dict, Type, Optional, Callable, Union
from lib.providers.base import ImageEditProvider


class ProviderRegistry:
    """
    Registry for managing image editing provider classes.

    This class provides a centralized way to register and retrieve provider
    implementations by their unique names. It follows the singleton pattern
    to ensure a single registry instance across the application.

    Attributes:
        _registry (Dict[str, Type[ImageEditProvider]]): Internal mapping of
            provider names to their class implementations.
    """

    _instance: Optional['ProviderRegistry'] = None
    _registry: Dict[str, Type[ImageEditProvider]] = {}

    def __new__(cls) -> 'ProviderRegistry':
        """
        Ensure singleton instance of the registry.

        Returns:
            ProviderRegistry: The singleton instance of the provider registry
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _register(self, provider_class: Type[ImageEditProvider], provider_name: Optional[str] = None) -> None:
        """
        Register a provider class in the registry.

        This internal method supports an optional provider_name to avoid
        instantiating the provider class at import time (which can require
        environment configuration). When provider_name is not provided, the
        provider class is instantiated to discover its name.

        Args:
            provider_class: The provider class to register, must be a subclass of ImageEditProvider
            provider_name: Optional explicit name to register under (avoids instantiation)

        Raises:
            TypeError: If provider_class is not a subclass of ImageEditProvider
            ValueError: If a provider with the same name is already registered
        """
        if not issubclass(provider_class, ImageEditProvider):
            raise TypeError(
                f"Provider class {provider_class.__name__} must be a subclass of ImageEditProvider"
            )

        if provider_name is None:
            # Derive provider name by instantiating (may require env); prefer explicit name when available
            try:
                resolved_name = provider_class().name
            except Exception as e:
                raise ValueError(
                    f"Failed to derive provider name for {provider_class.__name__}: {e}"
                ) from e
        else:
            resolved_name = provider_name

        if resolved_name in self._registry:
            raise ValueError(
                f"Provider with name '{resolved_name}' is already registered"
            )

        self._registry[resolved_name] = provider_class
        print(f"Registered provider: {resolved_name}")

    @classmethod
    def register(cls, provider_name: Optional[str] = None) -> Union[Callable[[Type[ImageEditProvider]], Type[ImageEditProvider]], Type[ImageEditProvider]]:
        """
        Class method decorator to register a provider class.

        Can be used as:
          @ProviderRegistry.register
          class MyProvider: ...

        Or with a custom name (avoids provider instantiation during import):
          @ProviderRegistry.register("custom_name")
          class MyProvider: ...

        Args:
            provider_name: Optional name to register the provider under.
                If None, uses the provider's natural name (may instantiate provider).

        Returns:
            Decorator function or decorated class
        """
        def decorator(provider_class: Type[ImageEditProvider]) -> Type[ImageEditProvider]:
            instance = cls()
            # Use internal registration to avoid instantiating when name provided
            instance._register(provider_class, provider_name)
            return provider_class

        # Handle both @register and @register("name") usage
        return decorator

    @classmethod
    def register_provider(cls, provider_class: Type[ImageEditProvider]) -> Type[ImageEditProvider]:
        """
        Class method to register a provider class (non-decorator form).

        Args:
            provider_class: The provider class to register

        Returns:
            Type[ImageEditProvider]: The provider class for decorator chaining
        """
        instance = cls()
        instance._register(provider_class)
        return provider_class

    def get_provider_class(self, name: str) -> Type[ImageEditProvider]:
        """
        Retrieve a provider class by its registered name.

        Args:
            name: The unique name of the provider to retrieve

        Returns:
            Type[ImageEditProvider]: The provider class implementation

        Raises:
            KeyError: If no provider with the given name is registered
        """
        if name not in self._registry:
            raise KeyError(
                f"Provider '{name}' not found in registry. "
                f"Available providers: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def list_providers(self) -> Dict[str, str]:
        """
        List all registered providers with their descriptions.

        Returns:
            Dict[str, str]: Mapping of provider names to their descriptions
        """
        return {
            name: provider_class().description
            for name, provider_class in self._registry.items()
        }

    def clear_registry(self) -> None:
        """
        Clear all registered providers from the registry.

        This is primarily useful for testing purposes.
        """
        self._registry.clear()

    @property
    def provider_count(self) -> int:
        """
        Get the number of registered providers.

        Returns:
            int: The count of registered providers
        """
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        """
        Check if a provider with the given name is registered.

        Args:
            name: The provider name to check

        Returns:
            bool: True if the provider is registered, False otherwise
        """
        return name in self._registry

    def __getitem__(self, name: str) -> Type[ImageEditProvider]:
        """
        Retrieve a provider class using dictionary-style access.

        Args:
            name: The provider name to retrieve

        Returns:
            Type[ImageEditProvider]: The provider class

        Raises:
            KeyError: If the provider is not found
        """
        return self.get_provider_class(name)

    def __iter__(self):
        """
        Iterate over registered provider names.

        Returns:
            iterator: Iterator over registered provider names
        """
        return iter(self._registry.keys())