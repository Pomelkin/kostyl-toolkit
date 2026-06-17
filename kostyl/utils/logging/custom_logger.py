from __future__ import annotations

from collections import namedtuple
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
except ImportError:

    def get_global_rank(process_group: ProcessGroup | None = None) -> int | None:
        """Gets the global rank of the current process in a distributed environment."""
        return None


try:
    from torch.nn.modules.module import _IncompatibleKeys
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

    def log_dist(
        self,
        level: str,
        msg: str,
        rank_only: int | None = None,
        process_group: ProcessGroup | None = None,
        *args,
        **kwargs: Any,
    ) -> None:
        """Log a distributed message, optionally limiting output to one global rank."""
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


def _log_dist(
    self: KostylLogger,
    level: str,
    msg: str,
    process_group: ProcessGroup | None = None,
    rank_only: int | None = None,
    *args,
    **kwargs: Any,
) -> None:
    global_rank = get_global_rank(process_group)  # ty:ignore[invalid-argument-type]

    if global_rank is not None:
        msg = f"[rank {global_rank}] {msg}"

    if rank_only is not None and global_rank is not None and global_rank != rank_only:
        return

    self.log(level, msg, *args, **kwargs)
    return


def _log_rank_zero(
    self: KostylLogger,
    level: str,
    msg: str,
    process_group: ProcessGroup | None = None,
    *args,
    **kwargs: Any,
) -> None:
    global_rank = get_global_rank(process_group) or 0  # ty:ignore[invalid-argument-type]
    if global_rank == 0:
        msg = f"[rank 0] {msg}"
        self.log(level, msg, *args, **kwargs)
    return


_base_logger = cast(KostylLogger, _base_logger)
_logger_cls = type(_base_logger)
_logger_cls.log_once = _log_once  # ty:ignore[invalid-assignment]
_logger_cls.warning_once = _warning_once  # ty:ignore[invalid-assignment]
_logger_cls.log_dist = _log_dist  # ty:ignore[invalid-assignment]
_logger_cls.log_rank_zero = _log_rank_zero  # ty:ignore[invalid-assignment]
