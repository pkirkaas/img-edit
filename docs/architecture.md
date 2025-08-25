# Architecture Overview

## Layers

The application is structured into several logical layers with comprehensive error handling:

- **IO**: [src/lib/image_io.py](src/lib/image_io.py) — read/write images, masks, format validation with rich error context
- **Config**: [src/lib/config.py](src/lib/config.py) — environment-backed settings with validation and error guidance, now with provider-specific configuration
- **Errors**: [src/lib/errors.py](src/lib/errors.py) — unified error hierarchy with context capture and user guidance
- **Providers**: [src/lib/providers/](src/lib/providers/) — provider abstraction layer with registry system for multiple AI service providers
- **Pipeline**: [src/lib/edit_pipeline.py](src/lib/edit_pipeline.py) — orchestrates provider calls and result handling with detailed error propagation
- **Logging**: [src/lib/logging_utils.py](src/lib/logging_utils.py) — structured logging utilities with JSON formatting for diagnostics
- **CLI**: [src/cli/main.py](src/cli/main.py) — Typer-based CLI with verbose/debug modes, provider selection, and error artifact saving

## Provider Abstraction Architecture

The application now features a provider-agnostic architecture that supports multiple AI image editing services through a unified interface.

### Provider Interface ([`ImageEditProvider`](src/lib/providers/base.py))
The [`ImageEditProvider`](src/lib/providers/base.py) abstract base class defines the common interface that all image editing providers must implement:

- **`name`**: Unique provider identifier (e.g., "huggingface", "replicate")
- **`description`**: Brief description of provider capabilities
- **`edit_image()`**: Core method for image editing with text instructions
- **`close()`**: Resource cleanup method

### Provider Registry System ([`ProviderRegistry`](src/lib/providers/registry.py))
The [`ProviderRegistry`](src/lib/providers/registry.py) implements a singleton pattern for managing provider implementations:

- **Dynamic registration**: Providers register themselves via decorators
- **Name-based lookup**: Retrieve providers by their unique names
- **Discovery**: List all available providers with descriptions
- **Validation**: Ensure provider classes implement the required interface

### Available Providers

- **Hugging Face Provider** ([`HuggingFaceProvider`](src/lib/providers/huggingface.py)): Uses Hugging Face's Inference API with support for models like Qwen/Qwen-Image-Edit. Wraps the existing HFImageEditClient for backward compatibility.

- **Replicate Provider** ([`ReplicateProvider`](src/lib/providers/replicate.py)): Uses Replicate's AI platform with support for various image editing models. Provides an alternative to Hugging Face with different pricing and performance characteristics.

### Provider Configuration ([`config.py`](src/lib/config.py))
The configuration system has been enhanced to support multiple providers:

- **Provider-specific settings**: Each provider can have its own API keys, timeouts, and model configurations
- **Default provider**: Configurable default provider (now set to "replicate")
- **Backward compatibility**: Automatic migration from legacy environment variables to provider configuration
- **Runtime selection**: CLI option to override the default provider per invocation

## Data Flow (Provider-based)

Image/Mask files -> [src/lib/image_io.py](src/lib/image_io.py) -> [src/lib/config.py](src/lib/config.py) -> [Provider Registry](src/lib/providers/registry.py) -> [Selected Provider](src/lib/providers/) -> [src/lib/edit_pipeline.py](src/lib/edit_pipeline.py) -> Output image via [src/lib/image_io.py](src/lib/image_io.py)

## Error Handling Architecture

The application features a comprehensive error handling system that works across all providers:

### Error Hierarchy
- **BaseError**: Root exception class for all application errors
- **ValidationError**: For input and configuration validation failures
- **ApiError**: For API-related errors (HTTP, provider responses) - now provider-agnostic
- **NetworkError**: For network connectivity issues
- **PipelineError**: For pipeline orchestration failures
- **IOError**: For file system and I/O operations
- **ProviderApiError**: Provider-specific API errors with rich context

### Context Capture
All errors include rich context information:
- Operation, provider, model, endpoint
- HTTP status, request ID, elapsed time
- Sanitized parameters, response snippets
- Root cause exceptions and user guidance
- Provider-specific error details

### Diagnostic Features
- **Verbose mode**: Detailed error messages with context and guidance
- **Debug mode**: Full error context saving to `tmp/last_error.json` and `tmp/last_error.txt`
- **Structured logging**: JSON-formatted logs in `logs/imgedit.log` with rotation
- **Artifact saving**: Success and error artifacts for comparison and debugging

## Error Flow

1. **Validation Errors**: Occur during input validation in image_io.py and config.py
2. **API Errors**: Occur during provider API calls in provider implementations
3. **Network Errors**: Occur during network operations in provider implementations
4. **Pipeline Errors**: Occur during orchestration in edit_pipeline.py
5. **CLI Handling**: Errors are captured, formatted, and presented with guidance in main.py

All errors propagate through the pipeline with increasing context and are finally handled by the CLI layer, which provides user-friendly messages and diagnostic artifacts.

## Backward Compatibility

The provider architecture maintains full backward compatibility with existing Hugging Face configurations:

- Legacy environment variables (HF_TOKEN, HF_API_TOKEN) are automatically converted to provider configuration
- Existing scripts using --endpoint or --model will continue to work
- The Hugging Face provider wraps the original HFImageEditClient for seamless transition
- Error handling and logging remain consistent across providers

## Documentation Governance

See [.roo/rules/running-docs.md](.roo/rules/running-docs.md) — docs must be kept up to date with implementation changes. For detailed error information and troubleshooting, see [docs/errors.md](./errors.md).