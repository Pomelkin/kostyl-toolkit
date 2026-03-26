from __future__ import annotations

import inspect
import os
import sys
import uuid
from collections import namedtuple
from copy import deepcopy
from functools import partialmethod
from pathlib import Path
from threading import Lock
from typing import Literal
from typing import cast

from loguru import Logger
from loguru import logger as _base_logger


class KostylLogger(Logger):  # noqa: D101
    def log_once(self, level: str, message: str, *args, **kwargs) -> None: ...  # noqa: ANN003, D102
    def warning_once(self, message: str, *args, **kwargs) -> None: ...  # noqa: ANN003, D102


try:
    from torch.nn.modules.module import (
        _IncompatibleKeys,  # pyright: ignore[reportAssignmentType]
    )
except Exception:

    class _IncompatibleKeys(
        namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"]),
    ):
        __slots__ = ()

        def __repr__(self) -> str:
            if not self.missing_keys and not self.unexpected_keys:
                return "<All keys matched successfully>"
            return super().__repr__()

        __str__ = __repr__

    _IncompatibleKeys = _IncompatibleKeys

_once_lock = Lock()
_once_keys: set[tuple[str, str]] = set()


def _log_once(self: KostylLogger, level: str, message: str, *args, **kwargs) -> None:  # noqa: ANN003
    key = (message, level)

    with _once_lock:
        if key in _once_keys:
            return
        _once_keys.add(key)

    self.log(level, message, *args, **kwargs)
    return


_base_logger = cast(KostylLogger, _base_logger)
_base_logger.log_once = _log_once
_base_logger.warning_once = partialmethod(_log_once, "WARNING")  # pyrefly: ignore


def _caller_filename() -> str:
    frame = inspect.stack()[2] if len(inspect.stack()) > 2 else inspect.stack()[1]
    name = Path(frame.filename).name
    return name


_DEFAULT_SINK_REMOVED = False
_DEFAULT_FMT = "<level>{level: <8}</level> {time:HH:mm:ss.SSS} [{extra[channel]}] <level>{message}</level>"
_ONLY_MESSAGE_FMT = "<level>{message}</level>"
_PRESETS = {"default": _DEFAULT_FMT, "only_message": _ONLY_MESSAGE_FMT}

KOSTYL_LOG_LEVEL = os.getenv("KOSTYL_LOG_LEVEL", "INFO")


def setup_logger(
    name: str | None = None,
    fmt: Literal["default", "only_message"] | str = "only_message",
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
            - "default": includes level, time, and channel
            - "only_message": outputs only the message itself
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
        fmt = _PRESETS[fmt]
    else:
        fmt = str(fmt)

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


def log_incompatible_keys(
    logger: Logger,
    incompatible_keys: _IncompatibleKeys
    | tuple[list[str], list[str]]
    | dict[str, list[str]],
    postfix_msg: str = "",
) -> None:
    """
    Logs warnings for incompatible keys encountered during model loading or state dict operations.

    Note: If incompatible_keys is of an unsupported type, an error message is logged and the function returns early.
    """
    incompatible_keys_: dict[str, list[str]] = {}
    match incompatible_keys:
        case (list() as missing_keys, list() as unexpected_keys):
            incompatible_keys_ = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
            }
        case _IncompatibleKeys() as ik:
            incompatible_keys_ = {
                "missing_keys": list(ik.missing_keys),
                "unexpected_keys": list(ik.unexpected_keys),
            }
        case dict() as d:
            incompatible_keys_ = deepcopy(d)
        case _:
            logger.error(
                f"Unsupported type for incompatible_keys: {type(incompatible_keys)}"
            )
            return

    for name, keys in incompatible_keys_.items():
        logger.warning(f"{name} {postfix_msg}: {', '.join(keys)}")
    return
