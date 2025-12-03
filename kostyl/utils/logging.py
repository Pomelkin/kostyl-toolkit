from __future__ import annotations

import inspect
import sys
import uuid
from collections import namedtuple
from copy import deepcopy
from functools import partialmethod
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

from loguru import logger as _base_logger


if TYPE_CHECKING:
    from loguru import Logger

    class CustomLogger(Logger):  # noqa: D101
        def log_once(self, level: str, message: str, *args, **kwargs) -> None: ...  # noqa: ANN003, D102
        def warning_once(self, message: str, *args, **kwargs) -> None: ...  # noqa: ANN003, D102
else:
    CustomLogger = type(_base_logger)

try:
    import torch.distributed as dist
    from torch.nn.modules.module import (
        _IncompatibleKeys,  # pyright: ignore[reportAssignmentType]
    )
except Exception:

    class _Dummy:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_initialized() -> bool:
            return False

        @staticmethod
        def get_rank() -> int:
            return 0

    class _IncompatibleKeys(
        namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"]),
    ):
        __slots__ = ()

        def __repr__(self) -> str:
            if not self.missing_keys and not self.unexpected_keys:
                return "<All keys matched successfully>"
            return super().__repr__()

        __str__ = __repr__

    dist = _Dummy()
    _IncompatibleKeys = _IncompatibleKeys

_once_lock = Lock()
_once_keys: set[tuple[str, str]] = set()


def _log_once(self: CustomLogger, level: str, message: str, *args, **kwargs) -> None:  # noqa: ANN003
    key = (message, level)

    with _once_lock:
        if key in _once_keys:
            return
        _once_keys.add(key)

    self.log(level, message, *args, **kwargs)
    return


_base_logger = cast(CustomLogger, _base_logger)
_base_logger.log_once = _log_once  # pyright: ignore[reportAttributeAccessIssue]
_base_logger.warning_once = partialmethod(_log_once, "WARNING")  # pyright: ignore[reportAttributeAccessIssue]


def _caller_filename() -> str:
    frame = inspect.stack()[2] if len(inspect.stack()) > 2 else inspect.stack()[1]
    name = Path(frame.filename).name
    return name


_DEFAULT_SINK_REMOVED = False
_DEFAULT_FMT = "<level>{level: <8}</level> {time:HH:mm:ss.SSS} [{extra[channel]}] <level>{message}</level>"
_ONLY_MESSAGE_FMT = "<level>{message}</level>"
_PRESETS = {"default": _DEFAULT_FMT, "only_message": _ONLY_MESSAGE_FMT}


def setup_logger(
    name: str | None = None,
    fmt: Literal["default", "only_message"] | str = "default",
    level: str = "INFO",
    add_rank: bool | None = None,
    sink=sys.stdout,
    colorize: bool = True,
    serialize: bool = False,
) -> CustomLogger:
    """
    Returns a bound logger with its own sink and formatting.

    Note: If name=None, the caller's filename (similar to __file__) is used automatically.

    Format example: "{level} {time:MM-DD HH:mm:ss} [{extra[channel]}] {message}"
    """
    global _DEFAULT_SINK_REMOVED
    if not _DEFAULT_SINK_REMOVED:
        _base_logger.remove()
        _DEFAULT_SINK_REMOVED = True

    if name is None:
        base = _caller_filename()
    else:
        base = name

    if (add_rank is None) or add_rank:
        try:
            add_rank = dist.is_available() and dist.is_initialized()
        except Exception:
            add_rank = False

    if add_rank:
        rank = dist.get_rank()
        channel = f"rank:{rank} - {base}"
    else:
        channel = base

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
    return cast(CustomLogger, logger)


def log_incompatible_keys(
    logger: Logger,
    incompatible_keys: _IncompatibleKeys
    | tuple[list[str], list[str]]
    | dict[str, list[str]],
    model_specific_msg: str = "",
) -> None:
    """
    Logs warnings for incompatible keys encountered during model loading or state dict operations.

    Note: If incompatible_keys is of an unsupported type, an error message is logged and the function returns early.

    Args:
        logger (Logger): The logger instance used to output warning messages.
        incompatible_keys (_IncompatibleKeys | tuple[list[str], list[str]] | dict[str, list[str]]): An object containing lists of missing and unexpected keys.
        model_specific_msg (str, optional): A custom message to append to the log output, typically
            indicating the model or context. Defaults to an empty string.

    Returns:
        None

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
        logger.warning(f"{name} {model_specific_msg}: {', '.join(keys)}")
    return
