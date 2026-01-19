# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Logging configuration for MPLang v2.

This module provides a unified logging setup for the MPLang library.
When MPLang is used as a library, logging is disabled by default (NullHandler),
allowing applications to configure logging as needed.

Example usage:
    >>> import mplang as mp
    >>> # Enable logging with INFO level
    >>> mp.setup_logging(level="INFO")
    >>> # Enable logging with DEBUG level and write to file
    >>> mp.setup_logging(level="DEBUG", filename="mplang.log")
    >>> # Customize format
    >>> mp.setup_logging(
    ...     level="DEBUG", format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    ... )
"""

import logging
import sys
from typing import Any, Literal

# Root logger for all MPLang v2 components
MPLANG_LOGGER_NAME = "mplang"

# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    format: str | None = None,
    date_format: str | None = None,
    filename: str | None = None,
    stream: Any = None,  # type: ignore[type-arg]
    force: bool = False,
    propagate: bool = False,
) -> None:
    """
    Configure logging for MPLang v2.

    This function sets up a logger for all MPLang v2 components. By default,
    MPLang uses a NullHandler to suppress log output when used as a library.
    Call this function to enable logging with custom settings.

    For unified application logging (application + mplang logs in same file):
    - Option 1: Configure Python's root logger, then call setup_logging(propagate=True)
    - Option 2: Just call setup_logging(filename="app.log") to add mplang logs to that file

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is INFO.
        format: Custom log format string. If None, uses default format.
        date_format: Custom date format string. If None, uses default format.
        filename: If provided, log to this file in addition to stream output.
        stream: Stream to log to (e.g., sys.stdout, sys.stderr). Default is sys.stderr.
                Set to False to disable stream output (only use file or propagation).
        force: If True, remove existing handlers before adding new ones.
        propagate: If True, allow logs to propagate to parent loggers (useful for
                   unified application logging). If False (default), mplang manages
                   its own handlers independently.

    Example:
        >>> import mplang as mp
        >>> # Enable debug logging to stderr
        >>> mp.setup_logging(level="DEBUG")
        >>> # Log mplang to file only
        >>> mp.setup_logging(level="INFO", filename="mplang.log", stream=False)
        >>>
        >>> # Unified application logging (recommended for applications)
        >>> import logging
        >>> logging.basicConfig(
        ...     level=logging.INFO,
        ...     filename="app.log",
        ...     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        ... )
        >>> mp.setup_logging(level="INFO", propagate=True)  # Inherits app config
    """
    # Get the root logger for mplang
    logger = logging.getLogger(MPLANG_LOGGER_NAME)

    # Set log level
    log_level = getattr(logging, level.upper())
    logger.setLevel(log_level)

    # If propagate=True, automatically remove NullHandler to avoid confusion
    # (NullHandler doesn't prevent propagation, but its presence is confusing)
    if propagate and not force:
        # Check if there's only a NullHandler
        if len(logger.handlers) == 1 and isinstance(
            logger.handlers[0], logging.NullHandler
        ):
            force = True  # Auto-enable force to remove NullHandler

    # Remove existing handlers if force=True
    if force:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # Set propagation behavior
    logger.propagate = propagate

    # If propagate is True and no handlers specified, just set level and propagate
    # This allows mplang logs to use the application's logging configuration
    if propagate and not filename and stream is None:
        return

    # Use default format if not provided
    log_format = format or DEFAULT_FORMAT
    log_date_format = date_format or DEFAULT_DATE_FORMAT
    formatter = logging.Formatter(log_format, datefmt=log_date_format)

    # Add stream handler (default to stderr, or skip if stream=False)
    if stream is not False:
        if stream is None:
            stream = sys.stderr
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Add file handler if filename is provided
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def disable_logging() -> None:
    """
    Disable all MPLang logging by adding a NullHandler.

    This is useful for testing or when you want to completely suppress
    MPLang log output.

    Example:
        >>> import mplang as mp
        >>> mp.disable_logging()
    """
    logger = logging.getLogger(MPLANG_LOGGER_NAME)
    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # Add NullHandler to suppress all output
    logger.addHandler(logging.NullHandler())
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific MPLang v2 module.

    This function should be used by all MPLang v2 modules to create their loggers.
    The logger name will be prefixed with 'mplang' automatically if not already.

    Args:
        name: Module name, typically __name__ from the calling module.

    Returns:
        A logger instance for the specified module.

    Example:
        >>> # In mplang/v2/edsl/tracer.py
        >>> logger = get_logger(__name__)  # Creates 'mplang.edsl.tracer' logger
    """
    # Ensure the logger name is under mplang hierarchy
    if not name.startswith(MPLANG_LOGGER_NAME):
        # Handle cases where __name__ might be a relative module name
        if name.startswith("mplang."):
            pass  # Already correct
        elif "." in name:
            # It's a submodule, but not under mplang, so prefix it.
            name = f"{MPLANG_LOGGER_NAME}.{name}"
        else:
            # Relative module name, prefix with mplang
            name = f"{MPLANG_LOGGER_NAME}.{name}"

    return logging.getLogger(name)


# Initialize with NullHandler by default (library mode)
# Applications using MPLang should call setup_logging() to enable logging
_root_logger = logging.getLogger(MPLANG_LOGGER_NAME)
if not _root_logger.handlers:
    _root_logger.addHandler(logging.NullHandler())
    _root_logger.propagate = False
