"""
Image editing pipeline orchestrator.

This module provides the main orchestration layer for the image editing application.
It coordinates the workflow between image I/O, Hugging Face API calls, and result handling.

The pipeline follows this workflow:
1. Load input image
2. Validate image and parameters
3. Send request to Hugging Face API
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
    PipelineError: Base exception for pipeline errors
    ImageValidationError: Exception for image validation failures
    PipelineTimeoutError: Exception for pipeline timeouts
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

from lib.config import Settings
from lib.hf_client import HFClient, HFAPIError, HFNetworkError
from lib.image_io import ImageIO, ImageValidationError
from lib.logging_utils import get_logger, log_timing, TimingContext

# Module-level logger
logger = get_logger(__name__)


class PipelineError(Exception):
    """Base exception for pipeline-related errors."""
    pass


class ImageValidationError(PipelineError):
    """Exception for image validation failures in the pipeline."""
    pass


class PipelineTimeoutError(PipelineError):
    """Exception for pipeline operation timeouts."""
    pass


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
        self.hf_client = HFClient(settings)
        
        logger.info("ImageEditPipeline initialized successfully")
    
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
            image_bytes, image_info = self._load_and_validate_image(input_path)
            
            # Execute the API request
            edited_image = self._execute_api_request(
                image_bytes, prompt, guidance_scale, strength, seed, **kwargs
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
            error_message = f"Pipeline failed: {e}"
            logger.error(error_message)
            
            return {
                "status": "error",
                "message": error_message,
                "input_path": input_path,
                "output_path": output_path,
                "execution_time": execution_time,
                "error_type": type(e).__name__
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
    
    def _load_and_validate_image(self, input_path: str) -> Tuple[bytes, Dict[str, any]]:
        """
        Load and validate the input image.
        
        Args:
            input_path: Path to input image
            
        Returns:
            Tuple of (image_bytes, image_info)
            
        Raises:
            ImageValidationError: If image validation fails
        """
        try:
            with TimingContext("image_loading", level=logging.DEBUG):
                image_bytes, image_info = self.image_io.load_image(input_path)
            
            logger.info(f"Loaded image: {input_path} ({image_info['format']}, {image_info['dimensions']})")
            return image_bytes, image_info
            
        except ImageValidationError as e:
            raise ImageValidationError(f"Failed to load/validate image: {e}")
        except Exception as e:
            raise ImageValidationError(f"Unexpected error loading image: {e}")
    
    def _execute_api_request(
        self,
        image_bytes: bytes,
        prompt: str,
        guidance_scale: Optional[float],
        strength: Optional[float],
        seed: Optional[int],
        **kwargs
    ) -> "Image.Image":
        """
        Execute the Hugging Face API request.
        
        Args:
            image_bytes: Raw image bytes
            prompt: Text prompt
            guidance_scale: Guidance scale parameter
            strength: Strength parameter
            seed: Optional seed
            **kwargs: Additional parameters
            
        Returns:
            PIL.Image.Image: Edited image from API
            
        Raises:
            HFAPIError: If API returns an error
            HFNetworkError: If network issues occur
            PipelineTimeoutError: If operation times out
        """
        try:
            with TimingContext("api_request", level=logging.INFO):
                edited_image = self.hf_client.inference_request(
                    image_bytes=image_bytes,
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    seed=seed,
                    **kwargs
                )
            
            logger.info(f"API request completed successfully")
            return edited_image
            
        except HFAPIError as e:
            logger.error(f"API error: {e}")
            raise
        except HFNetworkError as e:
            logger.error(f"Network error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected API error: {e}")
            raise PipelineError(f"API request failed: {e}")
    
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
            logger.error(f"Failed to save image: {e}")
            raise IOError(f"Could not save output image: {e}")
    
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
        self.hf_client.close()
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