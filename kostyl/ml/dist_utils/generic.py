import math
import os
from collections.abc import Callable
from functools import wraps
from typing import Literal
from typing import ParamSpec

import torch.distributed as dist

from kostyl.utils.logging import setup_logger


module_logger = setup_logger(fmt="default", name="dist_utils.generic")


def scale_lrs_by_world_size(
    lrs: dict[str, float],
    group: dist.ProcessGroup | None = None,
    inv_scale: bool = False,
    verbosity_level: Literal["rank-zero-only", "world"] | None = None,
) -> dict[str, float]:
    """
    Scale learning-rate configuration values to match the active distributed world size.

    Note:
        The value in the `lrs` will be modified in place.

    Args:
        lrs (dict[str, float]): A dictionary of learning rate names and their corresponding values to be scaled.
        group (dist.ProcessGroup | None): Optional process group used to determine
            the target world size. Defaults to the global process group.
        inv_scale (bool): If True, use the inverse square-root scale factor.
        verbosity_level (Literal["rank-zero-only", "world"] | None): Verbosity level for logging scaled values.
            - "rank-zero-only": Log only from the main process (rank 0).
            - "world": Log from all processes in the distributed environment.
            -  None: No logging.

    Returns:
        dict[str, float]: The learning-rate configuration with scaled values.

    """
    world_size = dist.get_world_size(group=group)

    scale = math.sqrt(world_size)
    if inv_scale:
        scale = scale**-1

    for name, value in lrs.items():
        old_value = value
        new_value = value * scale
        if verbosity_level is not None:
            msg = f"NEW lr {name.upper()}: {new_value}; OLD: {old_value}"
            if verbosity_level == "rank-zero-only":
                module_logger.log_rank_zero(
                    level="INFO",
                    msg=msg,
                    process_group=group,
                )
            elif verbosity_level == "world":
                # The rank tag is now prepended automatically by the logger.
                module_logger.info(msg)
        lrs[name] = new_value
    return lrs


def get_local_rank() -> int | None:
    """Gets the local rank of the current process in a distributed environment."""
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    # None to differentiate whether an environment variable was set at all
    return None


def is_local_rank_zero() -> bool:
    """Checks if the current process is the main process (rank 0) for the local node in a distributed environment."""
    local_rank = get_local_rank() or 0
    return local_rank == 0


def get_global_rank(group: dist.ProcessGroup | None = None) -> int | None:
    """Gets the global rank of the current process in a distributed environment."""
    if dist.is_initialized():
        return dist.get_rank(group=group)
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None


P = ParamSpec("P")


def local_rank_zero_only(func: Callable[P, None]) -> Callable[P, None]:
    """Decorator to ensure a function is executed only on the process with local rank 0 in a distributed environment."""

    @wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        if (get_local_rank() or 0) != 0:
            return None

        result = func(*args, **kwargs)
        if result is not None:
            raise RuntimeError(
                f"Function {func.__name__} must return only None, but got {type(result)}"
            )
        return None

    return _wrapper
