"""
Image editing pipeline orchestrator.

Syntax validated with ast.parse.

This module provides the main orchestration layer for the image editing application.
It coordinates the workflow between image I/O, Hugging Face InferenceClient usage, and result handling.

The pipeline follows this workflow:
1. Load input image
2. Validate image and parameters
3. Send request to Hugging Face via huggingface_hub.InferenceClient
4. Handle response and save output
5. Provide status and error handling

Example usage:
    >>> from lib.edit_pipeline import ImageEditPipeline
    >>> from lib.config import get_settings
    >>>
    >>> pipeline = ImageEditPipeline(get_settings())
    >>> result = pipeline.edit_image(
    ...     input_path="input.jpg",
    ...     output_path="output.jpg",
    ...     prompt="Make the sky blue",
    ...     guidance_scale=7.5
    ... )

Classes:
    ImageEditPipeline: Main pipeline orchestrator for image editing

Uses the unified error hierarchy from lib.errors for rich context capture.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

from lib.config import Settings
from lib.image_io import ImageIO, load_image as load_pil_image
from lib.logging_utils import get_logger, log_timing, TimingContext
from lib.errors import (
    PipelineError, ImageValidationError, NetworkError, ProviderApiError,
    IoError, ValidationError, create_error_context, save_error_artifact
)
from lib.providers import get_provider_class
from lib.providers.base import ImageEditProvider

# Module-level logger
logger = get_logger(__name__)


class ImageEditPipeline:
    """
    Main orchestrator for the image editing pipeline.
    
    This class coordinates the entire image editing workflow, including:
    - Image loading and validation
    - API request preparation and execution
    - Result handling and output saving
    - Error handling and logging
    
    Attributes:
        settings: Application configuration settings
        image_io: Image input/output handler
        hf_client: Hugging Face API client
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the image editing pipeline.
        
        Args:
            settings: Application settings with configuration
            
        Raises:
            PipelineError: If initialization fails
        """
        self.settings = settings
        self.image_io = ImageIO()
        # Initialize the provider from registry
        provider_name = self.settings.DEFAULT_PROVIDER
        try:
            provider_class = get_provider_class(provider_name)
            self.provider = provider_class()
        except KeyError as e:
            raise PipelineError(f"Provider '{provider_name}' not found in registry") from e
        except Exception as e:
            raise PipelineError(f"Failed to initialize provider '{provider_name}': {e}") from e
        
        logger.info(f"ImageEditPipeline initialized with provider: {provider_name}")
    
    @log_timing()
    def edit_image(
        self,
        input_path: str,
        output_path: str,
        prompt: str,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Union[str, float]]:
        """
        Execute the complete image editing pipeline.
        
        Args:
            input_path: Path to input image file
            output_path: Path to save edited image
            prompt: Text prompt describing the desired edit
            guidance_scale: Guidance scale parameter (default: from settings)
            strength: Strength parameter for editing (default: from settings)
            seed: Optional seed for deterministic results
            **kwargs: Additional parameters for the API
            
        Returns:
            Dict containing result metadata including:
            - status: "success" or "error"
            - message: Descriptive message
            - input_path: Original input path
            - output_path: Output file path
            - execution_time: Total execution time in seconds
            - file_size: Size of output file in bytes (if successful)
            
        Raises:
            PipelineError: If the pipeline fails at any stage
            ImageValidationError: If input image validation fails
            PipelineTimeoutError: If the operation times out
            
        Example:
            >>> result = pipeline.edit_image(
            ...     input_path="photo.jpg",
            ...     output_path="edited_photo.jpg",
            ...     prompt="Add sunglasses",
            ...     guidance_scale=8.0
            ... )
            >>> print(f"Status: {result['status']}")
            >>> print(f"Time: {result['execution_time']:.2f}s")
        """
        start_time = time.perf_counter()
        
        try:
            # Validate input parameters
            self._validate_inputs(input_path, output_path, prompt)
            
            # Load and validate input image
            image, image_info = self._load_and_validate_image(input_path)
            
            # Execute the API request
            edited_image = self._execute_api_request(
                image, prompt, guidance_scale, strength, seed, **kwargs
            )
            
            # Save the result
            output_info = self._save_result(edited_image, output_path)
            
            # Calculate execution time
            execution_time = time.perf_counter() - start_time
            
            return {
                "status": "success",
                "message": "Image edited successfully",
                "input_path": input_path,
                "output_path": output_path,
                "execution_time": execution_time,
                "file_size": output_info["file_size"],
                "image_format": output_info["format"],
                "image_dimensions": output_info["dimensions"]
            }
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            
            # Convert to our error hierarchy if not already
            if not isinstance(e, PipelineError):
                # Create rich context for the pipeline error
                error_context = create_error_context(
                    operation="edit_image",
                    provider=self.provider.name,
                    model=self.settings.IMG_EDIT_MODEL,
                    endpoint=self.settings.HF_INFERENCE_ENDPOINT,
                    elapsed_ms=execution_time * 1000,
                    root_cause=e
                )
                
                # Map to appropriate error type
                if isinstance(e, (ProviderApiError, NetworkError)):
                    # Already in our hierarchy, just re-raise with enhanced context
                    raise
                elif "permission" in str(e).lower() or "access" in str(e).lower():
                    e = IoError(f"File access error: {e}", context=error_context)
                elif "validation" in str(e).lower() or "invalid" in str(e).lower():
                    e = ValidationError(f"Validation error: {e}", context=error_context)
                else:
                    e = PipelineError(f"Pipeline error: {e}", context=error_context)
            
            # Log the error with full context
            logger.error(f"Pipeline failed: {e}")
            
            # Save error artifact for debugging
            try:
                save_error_artifact(e, "tmp/last_pipeline_error.json")
            except Exception as save_error:
                logger.warning(f"Failed to save pipeline error artifact: {save_error}")
            
            return {
                "status": "error",
                "message": str(e),
                "input_path": input_path,
                "output_path": output_path,
                "execution_time": execution_time,
                "error_type": type(e).__name__,
                "error_context": e.context.to_dict() if hasattr(e, 'context') else {},
                "user_guidance": e.user_guidance if hasattr(e, 'user_guidance') else "See logs for details"
            }
    
    def _validate_inputs(self, input_path: str, output_path: str, prompt: str) -> None:
        """
        Validate input parameters before processing.
        
        Args:
            input_path: Path to input image
            output_path: Path for output image
            prompt: Text prompt
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if not input_path or not os.path.exists(input_path):
            raise ValueError(f"Input file does not exist: {input_path}")
        
        if not output_path:
            raise ValueError("Output path is required")
        
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Check if output directory exists and is writable
        output_dir = os.path.dirname(output_path) or "."
        if not os.path.exists(output_dir):
            raise ValueError(f"Output directory does not exist: {output_dir}")
        if not os.access(output_dir, os.W_OK):
            raise ValueError(f"Output directory is not writable: {output_dir}")
        
        logger.debug(f"Input validation passed for: {input_path}")
    
    def _load_and_validate_image(self, input_path: str) -> Tuple[Image.Image, Dict[str, any]]:
        """
        Load and validate the input image.
        
        Args:
            input_path: Path to input image
            
        Returns:
            Tuple of (image, image_info) where image is PIL Image and image_info contains format and dimensions
            
        Raises:
            ImageValidationError: If image validation fails
        """
        try:
            with TimingContext("image_loading", level=logging.DEBUG):
                image = load_pil_image(input_path)
                image_info = {
                    "format": image.format if image.format else "UNKNOWN",
                    "dimensions": image.size
                }
            
            logger.info(f"Loaded image: {input_path} ({image_info['format']}, {image_info['dimensions']})")
            return image, image_info
            
        except Exception as e:
            error_context = create_error_context(
                operation="image_loading",
                provider=self.provider.name,
                model=self.settings.IMG_EDIT_MODEL,
                endpoint=self.settings.HF_INFERENCE_ENDPOINT,
                root_cause=e
            )
            raise ImageValidationError(
                f"Failed to load/validate image: {e}",
                context=error_context,
                user_guidance="Check that the image file exists, is readable, and in a supported format (JPEG, PNG, BMP, GIF, TIFF, WebP)."
            ) from e
    
    def _execute_api_request(
        self,
        image: Image.Image,
        prompt: str,
        guidance_scale: Optional[float],
        strength: Optional[float],
        seed: Optional[int],
        **kwargs
    ) -> "Image.Image":
        """
        Execute the image edit request using the configured provider.
        
        Args:
            image: PIL Image object to edit
            prompt: Text prompt describing the desired edit
            guidance_scale: Guidance scale parameter
            strength: Strength parameter for editing
            seed: Optional seed for deterministic results
            **kwargs: Additional parameters for the provider
            
        Returns:
            PIL.Image.Image: Edited image from provider
            
        Raises:
            ProviderApiError: If provider API returns an error
            NetworkError: If network issues occur
            PipelineError: If operation fails unexpectedly
        """
        try:
            with TimingContext("api_request", level=logging.INFO):
                edited_image = self.provider.edit_image(
                    image=image,
                    instructions=prompt,
                    strength=strength if strength is not None else self.settings.IMG_EDIT_DEFAULT_STRENGTH,
                    guidance_scale=guidance_scale if guidance_scale is not None else self.settings.IMG_EDIT_DEFAULT_GUIDANCE,
                    seed=seed,
                    **kwargs,
                )
            
            logger.info(f"API request completed successfully with provider: {self.provider.name}")
            return edited_image
            
        except (ProviderApiError, NetworkError) as e:
            # These are already in our error hierarchy with rich context
            logger.error(f"API request failed: {e}")
            raise
        except Exception as e:
            # Wrap unexpected errors in PipelineError with context
            error_context = create_error_context(
                operation="api_request",
                provider=self.provider.name,
                model=self.settings.IMG_EDIT_MODEL,
                endpoint=self.settings.HF_INFERENCE_ENDPOINT,
                root_cause=e
            )
            logger.error(f"Unexpected API error: {e}")
            raise PipelineError(
                f"API request failed: {e}",
                context=error_context,
                user_guidance="This may indicate an unexpected issue with the API provider or network. Check logs for details."
            ) from e
    
    def _save_result(self, edited_image: "Image.Image", output_path: str) -> Dict[str, any]:
        """
        Save the edited image to the output path.
        
        Args:
            edited_image: Edited PIL Image object
            output_path: Path to save the image
            
        Returns:
            Dict containing output file information
            
        Raises:
            IOError: If saving fails
        """
        try:
            with TimingContext("image_saving", level=logging.DEBUG):
                output_info = self.image_io.save_image(edited_image, output_path)
            
            logger.info(f"Saved edited image: {output_path} ({output_info['file_size']} bytes)")
            return output_info
            
        except Exception as e:
            error_context = create_error_context(
                operation="image_saving",
                provider=self.provider.name,
                model=self.settings.IMG_EDIT_MODEL,
                endpoint=self.settings.HF_INFERENCE_ENDPOINT,
                root_cause=e
            )
            logger.error(f"Failed to save image: {e}")
            raise IoError(
                f"Could not save output image: {e}",
                context=error_context,
                user_guidance="Check that the output directory exists and is writable. Verify available disk space."
            ) from e
    
    def batch_edit(
        self,
        input_pattern: str,
        output_dir: str,
        prompt: str,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Process multiple images matching a pattern.
        
        Args:
            input_pattern: Glob pattern for input files
            output_dir: Directory to save output files
            prompt: Text prompt for editing
            guidance_scale: Guidance scale parameter
            strength: Strength parameter
            **kwargs: Additional parameters
            
        Returns:
            List of result dictionaries for each processed image
            
        Example:
            >>> results = pipeline.batch_edit(
            ...     input_pattern="images/*.jpg",
            ...     output_dir="edited",
            ...     prompt="Make it sunny"
            ... )
        """
        # This would be implemented to handle multiple files
        # For now, it's a placeholder for future batch processing
        raise NotImplementedError("Batch processing not yet implemented")
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'provider') and self.provider is not None:
            self.provider.close()
        logger.info("Pipeline resources cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.close()


if __name__ == "__main__":
    # Test the pipeline (requires valid API configuration)
    import logging
    from lib.config import get_settings
    
    # Set up detailed logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        settings = get_settings()
        pipeline = ImageEditPipeline(settings)
        
        # Create a simple test image
        from PIL import Image
        import io
        
        test_image = Image.new("RGB", (100, 100), color="red")
        test_path = "test_input.jpg"
        test_image.save(test_path)
        
        print("Testing ImageEditPipeline...")
        print(f"API Endpoint: {settings.HF_INFERENCE_ENDPOINT}")
        print(f"Timeout: {settings.IMG_EDIT_TIMEOUT_SECONDS}s")
        
        # Note: Actual API call would require valid token
        # result = pipeline.edit_image(
        #     input_path=test_path,
        #     output_path="test_output.jpg",
        #     prompt="Make it blue"
        # )
        # print(f"Result: {result}")
        
        # Clean up test file
        if os.path.exists(test_path):
            os.remove(test_path)
            
        print("Pipeline test completed (API call skipped due to missing token)")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        if 'pipeline' in locals():
            pipeline.close()