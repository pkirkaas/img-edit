"""
Logging utilities for the image editing application.

This module provides centralized logging configuration and utility functions
for consistent logging across all application components. It includes a timing
decorator for performance monitoring and structured logging setup.

Features:
- Structured JSON logging for errors with rich context
- Human-readable console output
- File logging with rotation support
- Error-specific formatting with detailed diagnostics

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
    setup_structured_logging: Configure JSON-structured logging for errors
"""

import json
import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

# Type variable for generic function wrapping
F = TypeVar("F", bound=Callable[..., Any])

# Default logging format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# JSON log format for structured logging
JSON_LOG_FORMAT = "%(message)s"

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Default log file settings
DEFAULT_LOG_FILE = "logs/imgedit.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5


class StructuredFormatter(logging.Formatter):
    """
    Formatter for structured JSON logging of error messages.
    
    This formatter converts log records with error context into JSON format
    for better machine readability and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified record as JSON if it contains structured data.
        
        Args:
            record: Log record to format
            
        Returns:
            str: JSON string for structured data, or regular format for other messages
        """
        # Check if this is an error with structured context
        if (hasattr(record, 'structured_data') and
            record.levelno >= logging.ERROR and
            isinstance(record.structured_data, dict)):
            
            # Create structured log entry
            structured_log = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "data": record.structured_data
            }
            
            # Add exception info if available
            if record.exc_info:
                structured_log["exception"] = self.formatException(record.exc_info)
            
            return json.dumps(structured_log, default=str)
        
        # Fall back to default formatting for non-error or non-structured messages
        return super().format(record)


def setup_logging(
    level: int = DEFAULT_LOG_LEVEL,
    format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    filename: Optional[str] = None,
    max_size: int = MAX_LOG_SIZE,
    backup_count: int = BACKUP_COUNT
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
        max_size: Maximum log file size in bytes before rotation
        backup_count: Number of backup files to keep
        
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
            # Ensure log directory exists
            from pathlib import Path
            log_path = Path(filename)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use RotatingFileHandler for log rotation
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                filename,
                maxBytes=max_size,
                backupCount=backup_count,
                encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except (IOError, PermissionError, OSError) as e:
            root_logger.error(f"Failed to create file handler for {filename}: {e}")
    
    root_logger.info(f"Logging configured with level {logging.getLevelName(level)}")


def setup_structured_logging(
    level: int = DEFAULT_LOG_LEVEL,
    filename: str = DEFAULT_LOG_FILE,
    max_size: int = MAX_LOG_SIZE,
    backup_count: int = BACKUP_COUNT
) -> None:
    """
    Configure structured JSON logging for error diagnostics.
    
    This setup is specifically designed for error reporting and includes
    JSON formatting for machine-readable log analysis.
    
    Args:
        level: Logging level
        filename: Log file path for structured logs
        max_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Clear existing handlers
    logging.getLogger().handlers.clear()
    
    # Create structured formatter for errors
    structured_formatter = StructuredFormatter(JSON_LOG_FORMAT)
    
    # Console handler with regular formatting
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    
    # File handler with structured formatting
    try:
        from pathlib import Path
        from logging.handlers import RotatingFileHandler
        
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            filename,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(structured_formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(level)
        
        root_logger.info(f"Structured logging configured with level {logging.getLevelName(level)}")
        
    except (IOError, PermissionError, OSError) as e:
        # Fall back to console-only logging if file logging fails
        root_logger = logging.getLogger()
        root_logger.addHandler(console_handler)
        root_logger.setLevel(level)
        root_logger.error(f"Failed to configure structured file logging: {e}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance for the specified module.
    
    This function returns a logger instance with the given name, ensuring
    it inherits the root logger's configuration and supports structured logging.
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        logging.Logger: Configured logger instance with structured logging support
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Debug message")
        >>> logger.info("Information message")
        >>> logger.warning("Warning message")
        >>> logger.error("Error message")
        >>> logger.critical("Critical message")
        
        # Structured error logging
        >>> try:
        ...     raise ValueError("Test error")
        ... except Exception as e:
        ...     logger.error("Operation failed", extra={"structured_data": {"error": str(e)}})
    """
    logger = logging.getLogger(name)
    return logger


def log_error_with_context(
    logger: logging.Logger,
    message: str,
    error_context: Dict[str, Any],
    exc_info: Optional[Exception] = None
) -> None:
    """
    Log an error with structured context data for better diagnostics.
    
    This helper function ensures consistent error logging with rich context
    that can be parsed by monitoring systems.
    
    Args:
        logger: Logger instance to use
        message: Error message
        error_context: Dictionary with error context (operation, provider, etc.)
        exc_info: Optional exception for stack trace inclusion
    """
    logger.error(
        message,
        extra={"structured_data": error_context},
        exc_info=exc_info
    )


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