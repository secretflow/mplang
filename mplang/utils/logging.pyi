
import logging
from typing import Any, Literal

MPLANG_LOGGER_NAME: str
DEFAULT_FORMAT: str
DEFAULT_DATE_FORMAT: str

def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    format: str | None = None,
    date_format: str | None = None,
    filename: str | None = None,
    stream: Any = None,
    force: bool = False,
    propagate: bool = False,
) -> None: ...
def disable_logging() -> None: ...
def get_logger(name: str) -> logging.Logger: ...
