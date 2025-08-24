# Error Handling and Diagnostics

This document describes the comprehensive error handling system implemented in the Image Edit application, including error hierarchy, context capture, troubleshooting guidance, and diagnostic features.

## Error Hierarchy

The application uses a structured error hierarchy for consistent error reporting:

### BaseError
- **BaseException**: Root exception class for all application errors
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

## Error Context Fields

All errors include rich context information for better diagnostics:

| Field | Description | Example |
|-------|-------------|---------|
| `operation` | The operation that failed | `"image_loading"`, `"api_request"` |
| `provider` | API provider name | `"fal-ai"`, `"huggingface"` |
| `model` | Model identifier | `"Qwen/Qwen-Image-Edit"` |
| `endpoint` | API endpoint URL | `"https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit"` |
| `http_status` | HTTP status code | `400`, `401`, `500` |
| `request_id` | Provider request ID | `"req_1234567890"` |
| `elapsed_ms` | Request duration in milliseconds | `14250` |
| `params` | Sanitized request parameters | `{"prompt": "Remove sunglasses", "strength": 0.8}` |
| `response_snippet` | Response body snippet | `{"error": "Invalid image format"}` |
| `root_cause` | Underlying exception | `ValueError("Unsupported format")` |
| `file_path` | File path involved | `"/path/to/image.jpg"` |
| `detected_format` | Detected image format | `"ICO"` |
| `supported_formats` | Supported formats list | `["JPEG", "PNG", "BMP", "GIF", "TIFF", "WEBP"]` |

## Troubleshooting Common Errors

### API Errors (HFAPIError)
**Symptoms**: `API error during image_to_image: 'images' | context=provider=fal-ai, model=Qwen/Qwen-Image-Edit`

**Possible Causes**:
- Invalid or missing Hugging Face token
- Insufficient API credits/quota
- Unsupported image format or size
- Invalid prompt content or length
- Model downtime or maintenance

**Resolution Steps**:
1. Verify Hugging Face token is set: `echo $HF_TOKEN` or check `.env` file
2. Check API credits on Hugging Face dashboard
3. Ensure image is in supported format (JPEG, PNG, BMP, GIF, TIFF, WebP)
4. Reduce prompt length if excessively long
5. Try a different provider or model if available

### Network Errors (HFNetworkError)
**Symptoms**: `Network error during image_to_image: timeout`, `connection refused`

**Resolution**:
- Check internet connectivity
- Verify firewall/proxy settings
- Increase timeout value with `--timeout` flag
- Retry the operation

### Configuration Errors (ConfigurationError)
**Symptoms**: `Configuration error: No Hugging Face token provided`

**Resolution**:
```bash
# Set token in environment
export HF_TOKEN=your_hugging_face_token

# Or create .env file
echo "HF_TOKEN=your_hugging_face_token" > .env
```

### Image Validation Errors (ImageValidationError)
**Symptoms**: `Unsupported image format: ICO`, `Permission denied`, `Image file not found`

**Resolution**:
- Convert image to supported format (JPEG, PNG recommended)
- Check file permissions: `chmod +r image.jpg`
- Verify file exists at specified path
- For corrupt images, try re-downloading or using a different image

## Diagnostic Features

### Verbose and Debug Output
Use CLI flags for detailed error information:

```bash
# Basic verbose output
pdm run img-edit -i input.jpg -o output.jpg -p "Edit prompt" --verbose

# Full debug mode with artifact saving
pdm run img-edit -i input.jpg -o output.jpg -p "Edit prompt" --debug
```

### Error Artifacts
In debug mode, error artifacts are saved to `tmp/` directory:

- `tmp/last_error.json`: Structured error information with full context
- `tmp/last_error.txt`: Human-readable error summary
- `tmp/last_success.json`: Success operation details for comparison

### Log Files
Structured JSON logs are written to `logs/imgedit.log` with rotation (10MB max, 5 backups).

Example log entry:
```json
{
  "timestamp": "2025-08-24T21:34:19.123Z",
  "level": "ERROR",
  "logger": "lib.edit_pipeline",
  "message": "API error during image_to_image: 'images'",
  "data": {
    "operation": "api_request",
    "provider": "fal-ai",
    "model": "Qwen/Qwen-Image-Edit",
    "elapsed_ms": 14420,
    "http_status": 400,
    "params": {
      "prompt": "Remove sunglasses",
      "strength": 0.8
    },
    "response_snippet": {"error": "Invalid image format"}
  }
}
```

## Environment Checks

Before reporting issues, verify:

1. **Token Configuration**:
   ```bash
   echo $HF_TOKEN  # Should show your token
   # or
   echo $HF_API_TOKEN  # Fallback token
   ```

2. **Image Requirements**:
   - Formats: JPEG, PNG, BMP, GIF, TIFF, WebP
   - Size: Reasonable dimensions (typically < 2048x2048)
   - File size: Typically < 10MB

3. **Network Connectivity**:
   ```bash
   curl -s https://api-inference.huggingface.co/models/Qwen/Qwen-Image-Edit > /dev/null && echo "API reachable"
   ```

## Getting Help

When seeking assistance, provide:

1. The exact command used
2. Full error output (use `--verbose` or `--debug`)
3. Contents of `tmp/last_error.json` if available
4. Environment details: Python version, OS, application version

## Error Recovery

The application includes automatic retry logic for transient network errors and provides clear guidance for resolvable configuration issues. For persistent errors, check the Hugging Face status page and ensure your account has sufficient credits.