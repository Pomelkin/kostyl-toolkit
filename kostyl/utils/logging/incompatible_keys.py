from collections import namedtuple
from copy import deepcopy

from .custom_logger import KostylLogger


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


def log_incompatible_keys(
    logger: KostylLogger,
    incompatible_keys: _IncompatibleKeys
    | tuple[list[str], list[str]]
    | dict[str, list[str]],
    postfix_msg: str = "",
) -> None:
    """
    Logs warnings for incompatible keys encountered during model loading or state dict operations.

    Note: If incompatible_keys is of an unsupported type, an error message is logged and the function returns early.
    """
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
