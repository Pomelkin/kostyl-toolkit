import inspect
import os
import sys
import uuid
from pathlib import Path
from typing import Literal
from typing import cast

from loguru import logger as _base_logger

from .custom_logger import KostylLogger


_DEFAULT_SINK_REMOVED = False
_DETAILED_FMT = "<level>{level: <8}</level> {time:HH:mm:ss.SSS} [{extra[channel]}] <level>{message}</level>"
_MESSAGE_ONLY_FMT = "<level>{message}</level>"
_PRESETS = {
    "default": _DETAILED_FMT,
    "detailed": _DETAILED_FMT,
    "message_only": _MESSAGE_ONLY_FMT,
    "only_message": _MESSAGE_ONLY_FMT,
}

KOSTYL_LOG_LEVEL = os.getenv("KOSTYL_LOG_LEVEL", "INFO")


def _caller_filename() -> str:
    frame = inspect.stack()[2] if len(inspect.stack()) > 2 else inspect.stack()[1]
    name = Path(frame.filename).name
    return name


def setup_logger(
    name: str | None = None,
    fmt: Literal[
        "message_only", "only_message", "detailed", "default"
    ] = "message_only",
    level: str | None = None,
    sink=sys.stdout,
    colorize: bool = True,
    serialize: bool = False,
) -> KostylLogger:
    """
    Creates and configures a logger with custom formatting and output.

    The function automatically removes the default sink on first call and creates
    an isolated logger with a unique identifier for message filtering.

    Args:
        name (str | None, optional): Logger channel name. If None, automatically
            uses the calling function's filename. Defaults to None.
        fmt (Literal["default", "only_message"] | str, optional): Log message format.
            Available presets:
            - "detailed": includes level, time, and channel
            - "message_only": outputs only the message itself
            Custom format strings are also supported. Defaults to "only_message".
        level (str | None, optional): Logging level (TRACE, DEBUG, INFO, SUCCESS,
            WARNING, ERROR, CRITICAL). If None, uses the KOSTYL_LOG_LEVEL environment
            variable or "INFO" by default. Defaults to None.
        sink: Output object for logs (file, sys.stdout, sys.stderr, etc.).
            Defaults to sys.stdout.
        colorize (bool, optional): Enable colored output formatting.
            Defaults to True.
        serialize (bool, optional): Serialize logs to JSON format.
            Defaults to False.

    Returns:
        CustomLogger: Configured logger instance with additional methods
            log_once() and warning_once().

    Example:
        >>> # Basic usage with automatic name detection
        >>> logger = setup_logger()
        >>> logger.info("Hello World")

        >>> # With custom name and level
        >>> logger = setup_logger(name="MyApp", level="DEBUG")

        >>> # With custom format
        >>> logger = setup_logger(
        ...     name="API",
        ...     fmt="{level} | {time:YYYY-MM-DD HH:mm:ss} | {message}"
        ... )

    """
    global _DEFAULT_SINK_REMOVED
    if not _DEFAULT_SINK_REMOVED:
        _base_logger.remove()
        _DEFAULT_SINK_REMOVED = True

    if level is None:
        if KOSTYL_LOG_LEVEL not in {
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }:
            level = "INFO"
        else:
            level = KOSTYL_LOG_LEVEL

    if name is None:
        channel = _caller_filename()
    else:
        channel = name

    if fmt in _PRESETS:
        fmt = _PRESETS[fmt]  # ty:ignore[invalid-assignment]
    else:
        fmt = str(fmt)  # ty:ignore[invalid-assignment]

    logger_id = uuid.uuid4().hex

    _base_logger.add(
        sink,
        level=level,
        format=fmt,
        colorize=colorize,
        serialize=serialize,
        filter=lambda r: r["extra"].get("logger_id") == logger_id,
    )
    logger = _base_logger.bind(logger_id=logger_id, channel=channel)
    return cast(KostylLogger, logger)
