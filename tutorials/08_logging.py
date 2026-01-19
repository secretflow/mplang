"""
MPLang v2 Logging Guide

This tutorial demonstrates how to configure and use logging in MPLang v2.

Topics covered:
1. Default behavior (library mode)
2. Basic logging configuration
3. Unified application logging
4. Advanced scenarios (rotating logs, structured logging)
5. Best practices
"""

import json
import logging
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path

import mplang as mp


# =============================================================================
# 1. Default Behavior (Library Mode)
# =============================================================================
def example_1_default_behavior():
    """Example 1: Default behavior - MPLang operates silently when used as a library"""
    print("=" * 70)
    print("Example 1: Default Behavior (Library Mode)")
    print("=" * 70)

    # By default, MPLang v2 operates in library mode with no logging output
    # This allows applications to control logging configuration completely
    with mp.Interpreter(name="Silent") as _:
        print("✓ Interpreter created successfully (no log output)")

    print(
        "\nNote: MPLang uses NullHandler by default, won't interfere with application logging"
    )
    print()


# =============================================================================
# 2. Basic Logging Configuration
# =============================================================================
def example_2_basic_logging():
    """Example 2: Enable basic logging"""
    print("=" * 70)
    print("Example 2: Basic Logging Configuration")
    print("=" * 70)

    # Enable INFO level logging
    print("\n▸ INFO level logging:")
    mp.setup_logging(level="INFO")

    with mp.Interpreter(name="InfoExample") as _:
        print("✓ MPLang operational logs are visible")

    mp.disable_logging()

    # Enable DEBUG level logging (more detailed)
    print("\n▸ DEBUG level logging:")
    mp.setup_logging(level="DEBUG")

    with mp.Interpreter(name="DebugExample") as _:
        print("✓ More detailed debug information is visible")

    mp.disable_logging()

    # Log to file
    print("\n▸ Log output to file:")
    log_file = "mplang_basic.log"
    mp.setup_logging(level="INFO", filename=log_file)

    with mp.Interpreter(name="FileExample") as _:
        print(f"✓ Logs written to {log_file}")

    mp.disable_logging()
    print()


# =============================================================================
# 3. Unified Application Logging (Recommended)
# =============================================================================
def example_3_unified_logging():
    """Example 3: Unified management of application and MPLang logs"""
    print("=" * 70)
    print("Example 3: Unified Application Logging (Recommended)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "unified_app.log"

        # Method 1: Configure Python root logger first, MPLang inherits automatically
        logging.basicConfig(
            level=logging.INFO,
            filename=str(log_file),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )

        # Create application logger
        app_logger = logging.getLogger("myapp")

        # Both application and MPLang logs write to same file
        app_logger.info("Application started")

        with mp.Interpreter(name="UnifiedExample") as _:
            app_logger.info("Created MPLang Interpreter")

        app_logger.info("Application completed")

        # Display log content
        log_content = log_file.read_text()
        print("\nUnified log file content:")
        print("-" * 70)
        for line in log_content.strip().split("\n")[:10]:  # Display first 10 lines
            print(line)
        print("-" * 70)

        print("\n✓ Both application and MPLang logs are in the same file")
        print("  - Application logs: myapp")
        print("  - MPLang logs: mplang.v2.*")

    print()


# =============================================================================
# 4. Using propagate Parameter for Unified Management
# =============================================================================
def example_4_propagate():
    """Example 4: Using propagate parameter to propagate MPLang logs to root logger"""
    print("=" * 70)
    print("Example 4: Unified Management with propagate")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "propagate.log"

        # Configure root logger first
        logging.basicConfig(
            level=logging.INFO,
            filename=str(log_file),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            force=True,
        )

        # Enable MPLang logging and propagate to root logger
        mp.setup_logging(level="INFO", propagate=True)

        app_logger = logging.getLogger("myapp")
        app_logger.info("Application with propagate started")

        with mp.Interpreter(name="PropagateExample") as _:
            app_logger.info("MPLang operation")

        mp.disable_logging()

        log_content = log_file.read_text()
        print("\nLog content:")
        print("-" * 70)
        print(log_content)
        print("-" * 70)
        print(
            "\n✓ propagate=True allows MPLang logs to propagate to application's root logger"
        )

    print()


# =============================================================================
# 5. Separate Log Management
# =============================================================================
def example_5_separate_logs():
    """Example 5: Separate management of application and MPLang logs"""
    print("=" * 70)
    print("Example 5: Separate Log Management")
    print("=" * 70)

    app_log = "app_only.log"
    mplang_log = "mplang_only.log"

    # Clean up old handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure application logging
    app_logger = logging.getLogger("myapp")
    app_logger.setLevel(logging.INFO)
    app_handler = logging.FileHandler(app_log, mode="w")
    app_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    app_logger.addHandler(app_handler)
    app_logger.propagate = False

    # Configure MPLang logging (separate file)
    mp.setup_logging(
        level="INFO", filename=mplang_log, stream=False, propagate=False, force=True
    )

    # Usage
    app_logger.info("Application log -> app_only.log")

    with mp.Interpreter(name="SeparateExample") as _:
        app_logger.info("Application continues")

    print(f"✓ Logs separated:")
    print(f"  - {app_log}: Application logs")
    print(f"  - {mplang_log}: MPLang logs")

    mp.disable_logging()
    print()


# =============================================================================
# 6. Production Environment Complete Example
# =============================================================================
def example_6_production():
    """Example 6: Complete production environment configuration"""
    print("=" * 70)
    print("Example 6: Production Environment Configuration")
    print("=" * 70)

    import sys

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "production.log"

        # Cleanup
        mp.disable_logging()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # File handler - records INFO+
        file_handler = logging.FileHandler(str(log_file), mode="w")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)-30s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Console handler - only shows WARNING+
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # MPLang uses application's logging configuration
        mp.setup_logging(level="INFO", propagate=True)

        # Application code
        app_logger = logging.getLogger("production.app")
        app_logger.info("Application started (file only)")
        app_logger.warning("This is a warning (file + console)")

        with mp.Interpreter(name="Production") as _:
            app_logger.info("Executing MPLang operation (file only)")

        app_logger.info("Application completed")

        # Display file logs
        log_content = log_file.read_text()
        print("\nFile log content:")
        print("-" * 70)
        for line in log_content.strip().split("\n")[:10]:
            print(line)
        print("-" * 70)

        print("\n✓ Production configuration:")
        print("  - File: INFO+ level")
        print("  - Console: only WARNING+ level")

        # Cleanup
        root_logger.removeHandler(file_handler)
        root_logger.removeHandler(console_handler)
        file_handler.close()

    mp.disable_logging()
    print()


# =============================================================================
# 7. Advanced Scenario - Rotating Logs
# =============================================================================
def example_7_rotating_logs():
    """Example 7: Rotating logs (for long-running applications)"""
    print("=" * 70)
    print("Example 7: Rotating Logs")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "rotating.log"

        # Configure rotating log handler
        handler = RotatingFileHandler(
            str(log_file),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,  # Keep 5 backups
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

        app_logger = logging.getLogger("myapp")
        app_logger.info("Application with rotating logs started")

        with mp.Interpreter(name="RotatingExample") as _:
            app_logger.info("MPLang operation in progress")

        app_logger.info("Logs will rotate automatically when reaching 10MB")

        print("✓ Rotating log configuration complete")
        print("  - Max single log file size: 10 MB")
        print("  - Number of backups retained: 5")

        # Cleanup
        root_logger.removeHandler(handler)
        handler.close()

    print()


# =============================================================================
# 8. Advanced Scenario - Structured Logging (JSON)
# =============================================================================
def example_8_json_logging():
    """Example 8: Structured logging (JSON format, facilitates log analysis)"""
    print("=" * 70)
    print("Example 8: Structured Logging (JSON Format)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "structured.log"

        # Custom JSON formatter
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": self.formatTime(record, self.datefmt),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                }
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_data, ensure_ascii=False)

        # Configure JSON logging
        handler = logging.FileHandler(str(log_file), mode="w")
        handler.setFormatter(JsonFormatter())

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

        app_logger = logging.getLogger("myapp")
        app_logger.info("Structured logging application started")

        with mp.Interpreter(name="StructuredExample") as _:
            app_logger.info("Executing MPLang operation")

        app_logger.info("Application completed")

        # Display JSON logs
        log_content = log_file.read_text()
        print("\nJSON log content (first 3 lines):")
        print("-" * 70)
        for line in log_content.strip().split("\n")[:3]:
            log_entry = json.loads(line)
            print(json.dumps(log_entry, indent=2, ensure_ascii=False))
        print("-" * 70)
        print(
            "\n✓ JSON format facilitates processing by log analysis tools (ELK, Splunk, etc.)"
        )

        # Cleanup
        root_logger.removeHandler(handler)
        handler.close()

    print()


# =============================================================================
# 9. Hierarchical Logging Control
# =============================================================================
def example_9_hierarchical_logging():
    """Example 9: Hierarchical logging control"""
    print("=" * 70)
    print("Example 9: Hierarchical Logging Control")
    print("=" * 70)

    print("""
MPLang v2 uses a hierarchical logger structure:

mplang.v2                           # Root logger
├── mplang.v2.edsl                  # EDSL components
│   ├── mplang.v2.edsl.tracer       # Tracer logs
│   ├── mplang.v2.edsl.interpreter  # Interpreter logs
│   └── mplang.v2.edsl.jit          # JIT logs
├── mplang.v2.backends              # Backend implementations
├── mplang.v2.dialects              # Dialect primitives
└── mplang.v2.runtime               # Runtime components

You can set different log levels for different modules:
""")

    # Set base logging to WARNING
    mp.setup_logging(level="WARNING")

    # But enable DEBUG for specific module
    logging.getLogger("mplang.v2.edsl.tracer").setLevel(logging.DEBUG)

    print("✓ Configuration example:")
    print("  - Global: WARNING")
    print("  - mplang.v2.edsl.tracer: DEBUG")
    print(
        "\nThis allows viewing detailed logs only for modules of interest, avoiding information overload"
    )

    mp.disable_logging()
    print()


# =============================================================================
# Best Practices Summary
# =============================================================================
def print_best_practices():
    """Best practices summary"""
    print("=" * 70)
    print("Best Practices Summary")
    print("=" * 70)
    print("""
📋 Choose appropriate logging configuration based on different scenarios:

1️⃣ [Recommended] Unified Management (Application + MPLang)
   Use case: Most application scenarios
   
   ```python
   import logging
   import mplang.v2 as mp

   # Configure Python root logger
   logging.basicConfig(
       level=logging.INFO,
       filename="app.log",
       format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
   )

   # MPLang logs automatically inherit configuration (Method 1)
   with mp.Interpreter() as interp:
       pass  # Logs automatically written to app.log
   
   # Or explicit propagation (Method 2)
   mp.setup_logging(level="INFO", propagate=True)
   ```

2️⃣ Only Log MPLang
   Use case: Only want to view MPLang internals
   
   ```python
   mp.setup_logging(
       level="INFO",
       filename="mplang.log",
       stream=False  # Don't output to console
   )
   ```

3️⃣ Separate Management (Application and Library)
   Use case: Need to analyze logs independently
   
   ```python
   # Application configures its own logger
   app_logger = logging.getLogger("myapp")
   app_logger.addHandler(...)

   # MPLang independent configuration
   mp.setup_logging(filename="mplang.log", propagate=False)
   ```

4️⃣ Production Environment (Multiple Handlers)
   Use case: File records detailed logs, console shows only important info
   
   ```python
   # File handler: INFO+
   file_handler = logging.FileHandler("app.log")
   file_handler.setLevel(logging.INFO)
   
   # Console handler: WARNING+
   console_handler = logging.StreamHandler()
   console_handler.setLevel(logging.WARNING)
   
   root_logger = logging.getLogger()
   root_logger.addHandler(file_handler)
   root_logger.addHandler(console_handler)
   
   mp.setup_logging(level="INFO", propagate=True)
   ```

🎯 Key Parameter Descriptions:
   - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   - filename: Log file path
   - propagate: Whether to propagate to parent logger (True=unified, False=independent)
   - stream: Whether to output to stderr (False=file only)
   - force: Whether to clear existing configuration and reset

📁 Recommended Log File Organization:
   - Development: ./logs/dev.log
   - Testing: ./logs/test.log
   - Production: /var/log/myapp/production.log

🔧 Environment Variable Control (Recommended):
   ```python
   import os
   log_level = os.getenv("LOG_LEVEL", "INFO")
   log_file = os.getenv("LOG_FILE", "app.log")
   mp.setup_logging(level=log_level, filename=log_file)
   ```

🐛 Debugging MPLang Internal Issues:
   ```python
   mp.setup_logging(level="DEBUG")  # View detailed execution flow
   ```

🔇 Silent Mode (Library Mode):
   ```python
   # Don't call setup_logging(), MPLang stays silent
   # Or explicitly disable
   mp.disable_logging()
   ```

✅ Verify Logging Configuration:
   ```python
   logger = logging.getLogger("mplang.v2")
   print(f"Logger level: {logger.level}")
   print(f"Handlers: {logger.handlers}")
   ```
""")


# =============================================================================
# Run All Examples
# =============================================================================
def main():
    """Run all examples"""
    example_1_default_behavior()
    example_2_basic_logging()
    example_3_unified_logging()
    example_4_propagate()
    example_5_separate_logs()
    example_6_production()
    example_7_rotating_logs()
    example_8_json_logging()
    example_9_hierarchical_logging()
    print_best_practices()

    print("\n" + "=" * 70)
    print("✅ All Examples Completed")
    print("=" * 70)
    print("""
Summary:
MPLang v2's logging system is designed as a good citizen of the standard Python
logging library, seamlessly integrating into any application's logging management.

Core Principles:
- Silent by default (library mode), doesn't interfere with applications
- Based on standard logging module, fully compatible
- Supports propagation mechanism for unified management
- Hierarchical structure, supports fine-grained control
    """)


if __name__ == "__main__":
    main()
