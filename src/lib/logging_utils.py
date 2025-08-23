"""
Logging utilities for the image editing application.

This module provides centralized logging configuration and utility functions
for consistent logging across all application components. It includes a timing
decorator for performance monitoring and structured logging setup.

Example usage:
    >>> from lib.logging_utils import get_logger, log_timing
    >>> logger = get_logger(__name__)
    >>> 
    >>> @log_timing
    ... def slow_function():
    ...     # Function implementation
    ...     pass

Functions:
    get_logger: Get a configured logger instance for a module
    log_timing: Decorator to log function execution time
    setup_logging: Configure the root logger with desired settings
"""

import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

# Type variable for generic function wrapping
F = TypeVar("F", bound=Callable[..., Any])

# Default logging format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO


def setup_logging(
    level: int = DEFAULT_LOG_LEVEL,
    format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    filename: Optional[str] = None
) -> None:
    """
    Configure the root logger with specified settings.
    
    This function sets up the logging configuration for the entire application.
    It can be called at application startup to ensure consistent logging.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        format: Log message format string
        date_format: Date format for timestamps
        filename: Optional file path for file logging. If None, logs to console only.
        
    Example:
        >>> setup_logging(level=logging.DEBUG, filename="app.log")
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format, date_format)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add console handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(level)
    
    # Add file handler if filename is provided
    if filename:
        try:
            file_handler = logging.FileHandler(filename, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except (IOError, PermissionError) as e:
            root_logger.error(f"Failed to create file handler for {filename}: {e}")
    
    root_logger.info(f"Logging configured with level {logging.getLevelName(level)}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance for the specified module.
    
    This function returns a logger instance with the given name, ensuring
    it inherits the root logger's configuration.
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Debug message")
        >>> logger.info("Information message")
        >>> logger.warning("Warning message")
        >>> logger.error("Error message")
        >>> logger.critical("Critical message")
    """
    logger = logging.getLogger(name)
    return logger


def log_timing(level: int = logging.INFO) -> Callable[[F], F]:
    """
    Decorator factory to log function execution time.
    
    This decorator measures and logs the execution time of the decorated function.
    It can be configured with different log levels.
    
    Args:
        level: Log level for timing messages (default: logging.INFO)
        
    Returns:
        Callable: Decorator function
        
    Example:
        >>> @log_timing(level=logging.DEBUG)
        ... def expensive_operation():
        ...     time.sleep(1)
        ... 
        >>> expensive_operation()  # Logs: "expensive_operation executed in 1.00s"
        
        >>> @log_timing()
        ... def another_function():
        ...     pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                logger.log(
                    level,
                    f"{func.__name__} executed in {execution_time:.2f}s",
                    extra={"execution_time": execution_time}
                )
        
        return cast(F, wrapper)
    
    return decorator


class TimingContext:
    """
    Context manager for timing code blocks with logging.
    
    This context manager measures and logs the execution time of a code block.
    It provides more flexibility than the decorator for timing arbitrary code sections.
    
    Example usage:
        >>> with TimingContext("image_processing", level=logging.DEBUG):
        ...     # Code to time
        ...     process_image()
    """
    
    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize the timing context.
        
        Args:
            name: Name for the timed operation (appears in logs)
            level: Log level for timing message
        """
        self.name = name
        self.level = level
        self.logger = get_logger(__name__)
        self.start_time: Optional[float] = None
    
    def __enter__(self) -> "TimingContext":
        """Enter the context and start timing."""
        self.start_time = time.perf_counter()
        self.logger.log(self.level, f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context and log the elapsed time."""
        if self.start_time is not None:
            end_time = time.perf_counter()
            execution_time = end_time - self.start_time
            
            status = "completed" if exc_type is None else "failed"
            self.logger.log(
                self.level,
                f"{self.name} {status} in {execution_time:.2f}s",
                extra={"execution_time": execution_time, "status": status}
            )


# Set up default logging when module is imported
setup_logging()


if __name__ == "__main__":
    # Test the logging utilities
    logger = get_logger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test timing decorator
    @log_timing()
    def test_function():
        time.sleep(0.1)
        return "done"
    
    result = test_function()
    print(f"Function returned: {result}")
    
    # Test timing context
    with TimingContext("test_operation"):
        time.sleep(0.05)
    
    print("Logging utilities test completed successfully")