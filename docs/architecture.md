# Architecture Overview

## Layers

The application is structured into several logical layers with comprehensive error handling:

- **IO**: [src/lib/image_io.py](src/lib/image_io.py) — read/write images, masks, format validation with rich error context
- **Config**: [src/lib/config.py](src/lib/config.py) — environment-backed settings with validation and error guidance
- **Errors**: [src/lib/errors.py](src/lib/errors.py) — unified error hierarchy with context capture and user guidance
- **Client**: [src/lib/hf_client.py](src/lib/hf_client.py) — thin wrapper over huggingface_hub.InferenceClient with enhanced error parsing
- **Pipeline**: [src/lib/edit_pipeline.py](src/lib/edit_pipeline.py) — orchestrates client calls and result handling with detailed error propagation
- **Logging**: [src/lib/logging_utils.py](src/lib/logging_utils.py) — structured logging utilities with JSON formatting for diagnostics
- **CLI**: [src/cli/main.py](src/cli/main.py) — Typer-based CLI with verbose/debug modes and error artifact saving

## Data Flow (InferenceClient-based)

Image/Mask files -> [src/lib/image_io.py](src/lib/image_io.py) -> [src/lib/config.py](src/lib/config.py) -> [src/lib/hf_client.py](src/lib/hf_client.py) -> [src/lib/edit_pipeline.py](src/lib/edit_pipeline.py) -> Output image via [src/lib/image_io.py](src/lib/image_io.py)

## Error Handling Architecture

The application now features a comprehensive error handling system:

### Error Hierarchy
- **BaseError**: Root exception class for all application errors
- **ValidationError**: For input and configuration validation failures
- **ApiError**: For API-related errors (HTTP, provider responses)
- **NetworkError**: For network connectivity issues
- **PipelineError**: For pipeline orchestration failures
- **IOError**: For file system and I/O operations

### Specialized Errors
- **ImageValidationError**: Image format, corruption, or access issues
- **ConfigurationError**: Environment variable and configuration issues
- **HFAPIError**: Hugging Face API-specific errors
- **HFNetworkError**: Hugging Face network connectivity issues

### Context Capture
All errors include rich context information:
- Operation, provider, model, endpoint
- HTTP status, request ID, elapsed time
- Sanitized parameters, response snippets
- Root cause exceptions and user guidance

### Diagnostic Features
- **Verbose mode**: Detailed error messages with context and guidance
- **Debug mode**: Full error context saving to `tmp/last_error.json` and `tmp/last_error.txt`
- **Structured logging**: JSON-formatted logs in `logs/imgedit.log` with rotation
- **Artifact saving**: Success and error artifacts for comparison and debugging

## Error Flow

1. **Validation Errors**: Occur during input validation in image_io.py and config.py
2. **API Errors**: Occur during Hugging Face API calls in hf_client.py
3. **Network Errors**: Occur during network operations in hf_client.py
4. **Pipeline Errors**: Occur during orchestration in edit_pipeline.py
5. **CLI Handling**: Errors are captured, formatted, and presented with guidance in main.py

All errors propagate through the pipeline with increasing context and are finally handled by the CLI layer, which provides user-friendly messages and diagnostic artifacts.

## Documentation Governance

See [.roo/rules/running-docs.md](.roo/rules/running-docs.md) — docs must be kept up to date with implementation changes. For detailed error information and troubleshooting, see [docs/errors.md](./errors.md).