from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from loguru import logger as _base_logger


if TYPE_CHECKING:
    from loguru import Logger

    try:
        from torch.distributed import ProcessGroup
    except ImportError:

        class ProcessGroup:
            """Dummy ProcessGroup class for non-distributed environments."""

            pass
else:
    Logger = object

try:
    from kostyl.ml.dist_utils import get_global_rank
except Exception:

    def get_global_rank(group=None) -> int | None:
        """Gets the global rank of the current process in a distributed environment."""
        return None


class KostylLogger(Logger):  # noqa: D101
    def log_once(
        self,
        level: str,
        message: str,
        *args,
        **kwargs: Any,
    ) -> None:
        """Log a message with the specified level only once."""
        raise NotImplementedError(
            "This method is implemented dynamically and should not be called directly."
        )

    def warning_once(self, message: str, *args, **kwargs: Any) -> None:
        """Log a warning message only once."""
        raise NotImplementedError(
            "This method is implemented dynamically and should not be called directly."
        )

    def log_rank_zero(
        self,
        level: str,
        msg: str,
        process_group: ProcessGroup | None = None,
        *args,
        **kwargs: Any,
    ) -> None:
        """Log a message exclusively on global rank 0 (the main process)."""
        raise NotImplementedError(
            "This method is implemented dynamically and should not be called directly."
        )


_once_lock = Lock()
_once_keys: set[tuple[str, str]] = set()


def _log_once(
    self: KostylLogger, level: str, message: str, *args, **kwargs: Any
) -> None:
    key = (message, level)

    with _once_lock:
        if key in _once_keys:
            return
        _once_keys.add(key)

    self.log(level, message, *args, **kwargs)
    return


def _warning_once(self: KostylLogger, message: str, *args, **kwargs: Any) -> None:
    self.log_once("WARNING", message, *args, **kwargs)
    return


def _log_rank_zero(
    self: KostylLogger,
    level: str,
    msg: str,
    process_group: ProcessGroup | None = None,
    *args,
    **kwargs: Any,
) -> None:
    # ``None`` (distributed not initialized) is treated as the single-process
    # main rank so messages still surface when running without torchrun.
    if (get_global_rank(process_group) or 0) == 0:  # ty:ignore[invalid-argument-type]
        self.log(level, msg, *args, **kwargs)
    return


def _log_with_rank(
    self: KostylLogger,
    level: Any,
    from_decorator: bool,
    options: tuple,
    message: str,
    args: tuple,
    kwargs: dict,
) -> None:
    """Prefix every record with ``[RANK: <id>]`` while a process group is initialized."""
    global_rank = get_global_rank()
    if global_rank is not None:
        message = f"[RANK: {global_rank}] {message}"

    # This wrapper adds one frame; bump the recorded depth so loguru still
    # resolves the real caller for ``{name}``/``{function}``/``{line}``.
    exception, depth, *rest = options
    options = (exception, depth + 1, *rest)

    return _orig_log(self, level, from_decorator, options, message, args, kwargs)


_base_logger = cast(KostylLogger, _base_logger)
_logger_cls = type(_base_logger)
_orig_log = _logger_cls._log  # ty:ignore[unresolved-attribute]
_logger_cls.log_once = _log_once  # ty:ignore[invalid-assignment]
_logger_cls.warning_once = _warning_once  # ty:ignore[invalid-assignment]
_logger_cls.log_rank_zero = _log_rank_zero  # ty:ignore[invalid-assignment]
_logger_cls._log = _log_with_rank  # ty:ignore[unresolved-attribute]
