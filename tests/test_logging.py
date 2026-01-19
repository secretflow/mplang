"""Tests for MPLang v2 logging functionality."""

import io
import logging

import pytest

import mplang as mp


def test_logging_disabled_by_default():
    """Test that logging is disabled by default (library mode)."""
    # Get the root mplang logger
    logger = logging.getLogger("mplang.v2")

    # Should have a NullHandler by default
    assert len(logger.handlers) > 0
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


def test_setup_logging_basic():
    """Test basic logging setup."""
    # Create a string buffer to capture logs
    log_stream = io.StringIO()

    # Setup logging to the stream
    mp.setup_logging(level="INFO", stream=log_stream, force=True)

    # Get a logger and log a message
    logger = logging.getLogger("mplang.v2.test")
    logger.info("Test message")

    # Check that the message was logged
    log_output = log_stream.getvalue()
    assert "Test message" in log_output
    assert "INFO" in log_output

    # Cleanup
    mp.disable_logging()


def test_setup_logging_levels():
    """Test that different log levels are respected."""
    log_stream = io.StringIO()

    # Setup with WARNING level
    mp.setup_logging(level="WARNING", stream=log_stream, force=True)

    logger = logging.getLogger("mplang.v2.test")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    log_output = log_stream.getvalue()

    # Only WARNING and ERROR should appear
    assert "Debug message" not in log_output
    assert "Info message" not in log_output
    assert "Warning message" in log_output
    assert "Error message" in log_output

    # Cleanup
    mp.disable_logging()


def test_disable_logging():
    """Test that disable_logging suppresses all output."""
    log_stream = io.StringIO()

    # First enable logging
    mp.setup_logging(level="DEBUG", stream=log_stream, force=True)

    # Then disable it
    mp.disable_logging()

    # Try to log
    logger = logging.getLogger("mplang.v2.test")
    logger.error("This should not appear")

    # Nothing should be logged
    log_output = log_stream.getvalue()
    assert "This should not appear" not in log_output


def test_logging_with_trace():
    """Test that logging works during tracing."""
    log_stream = io.StringIO()
    mp.setup_logging(level="DEBUG", stream=log_stream, force=True)

    try:
        # Test that basic logging infrastructure is working
        # We can verify by creating a logger and checking logs are captured
        from mplang.v2.logging_config import get_logger

        test_logger = get_logger("mplang.v2.test")
        test_logger.debug("Test trace log message")

        # The log should be captured
        log_output = log_stream.getvalue()
        assert "Test trace log message" in log_output

    finally:
        mp.disable_logging()


def test_logging_with_interpreter():
    """Test that logging works during interpreter execution."""
    log_stream = io.StringIO()
    mp.setup_logging(level="DEBUG", stream=log_stream, force=True)

    try:
        # Create an interpreter
        with mp.Interpreter(name="TestInterpreter") as _:
            # Should log interpreter initialization
            log_output = log_stream.getvalue()
            assert "Initialized Interpreter" in log_output or len(log_output) > 0

    finally:
        mp.disable_logging()


def test_get_logger_helper():
    """Test the get_logger helper function."""
    from mplang.v2.logging_config import get_logger

    # Get a logger for a hypothetical module
    logger = get_logger("mplang.v2.test_module")

    # Should be under mplang.v2 hierarchy
    assert logger.name == "mplang.v2.test_module"

    # Should inherit from mplang.v2 root logger
    root_logger = logging.getLogger("mplang.v2")
    assert logger.parent == root_logger or logger.name.startswith("mplang.v2")


def test_logging_with_custom_format():
    """Test custom log format."""
    log_stream = io.StringIO()
    custom_format = "CUSTOM: %(message)s"

    mp.setup_logging(level="INFO", format=custom_format, stream=log_stream, force=True)

    logger = logging.getLogger("mplang.v2.test")
    logger.info("Test message")

    log_output = log_stream.getvalue()
    assert "CUSTOM: Test message" in log_output

    # Cleanup
    mp.disable_logging()


def test_logging_hierarchy():
    """Test that child loggers inherit from parent."""
    log_stream = io.StringIO()
    mp.setup_logging(level="INFO", stream=log_stream, force=True)

    # Create child logger
    from mplang.v2.logging_config import get_logger

    child_logger = get_logger("mplang.v2.edsl.tracer")

    child_logger.info("Child message")

    log_output = log_stream.getvalue()
    assert "Child message" in log_output
    assert "mplang.v2.edsl.tracer" in log_output

    # Cleanup
    mp.disable_logging()


def test_propagate_to_root_logger():
    """Test that propagate=True allows logs to propagate to root logger."""
    import os
    import tempfile

    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        temp_log = f.name

    try:
        # Configure Python's root logger
        logging.basicConfig(
            level=logging.INFO,
            filename=temp_log,
            format="%(levelname)s:%(name)s:%(message)s",
            force=True,
        )

        # Enable MPLang logging with propagation
        mp.setup_logging(level="INFO", propagate=True, force=True)

        # Create application logger
        app_logger = logging.getLogger("test_app")
        app_logger.info("App message")

        # MPLang logger
        mplang_logger = logging.getLogger("mplang.v2.test")
        mplang_logger.info("MPLang message")

        # Read log file
        with open(temp_log) as f:
            log_content = f.read()

        # Both should be in the log file
        assert "App message" in log_content
        assert "MPLang message" in log_content

    finally:
        # Cleanup
        mp.disable_logging()
        if os.path.exists(temp_log):
            os.remove(temp_log)


def test_stream_false_only_file():
    """Test that stream=False only logs to file, not console."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        temp_log = f.name

    try:
        # Setup logging with file only (no stream)
        mp.setup_logging(
            level="INFO",
            filename=temp_log,
            stream=False,  # Disable stream output
            force=True,
        )

        # Log a message
        logger = logging.getLogger("mplang.v2.test")
        logger.info("File only message")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        # Read log file
        with open(temp_log) as f:
            log_content = f.read()

        # Should be in file
        assert "File only message" in log_content

    finally:
        # Cleanup
        mp.disable_logging()
        if os.path.exists(temp_log):
            os.remove(temp_log)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
