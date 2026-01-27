import math
import os
from typing import Literal

import torch.distributed as dist

from kostyl.utils.logging import KostylLogger
from kostyl.utils.logging import setup_logger


module_logger = setup_logger()


def log_dist(
    msg: str,
    logger: KostylLogger | None = None,
    level: Literal["info", "warning", "error", "warning_once", "debug"] = "info",
    log_scope: Literal["only-zero-rank", "world"] = "world",
    group: dist.ProcessGroup | None = None,
) -> None:
    """
    Log a message in a distributed environment based on the specified verbosity level.

    Args:
        msg (str): The message to log.
        log_scope (Literal["only-zero-rank", "world"]): The verbosity level for logging.
            - "only-zero-rank": Log only from the main process (rank 0).
            - "world": Log from all processes in the distributed environment.
        logger (KostylLogger | None): The logger instance to use. If None, the module logger is used.
        level (Literal["info", "warning", "error", "warning_once", "debug"]): The logging level.
        group (dist.ProcessGroup | None): Optional process group used to determine ranks. Defaults to the global process group.

    """
    if logger is None:
        logger = module_logger

    log_attr = getattr(logger, level, None)
    if log_attr is None:
        raise ValueError(f"Invalid logging level: {level}")

    if not dist.is_initialized():
        module_logger.warning_once(
            "Distributed process group is not initialized. Logging from the current process only."
        )
        log_attr(msg)
        return

    match log_scope:
        case "only-zero-rank":
            if group is None:
                module_logger.debug(
                    "No process group provided; assuming global group for rank check."
                )
                group = dist.group.WORLD
            group_rank = dist.get_rank(group=group)
            if dist.get_global_rank(group=group, group_rank=group_rank) == 0:  # pyright: ignore[reportArgumentType]
                log_attr(msg)
        case "world":
            log_attr(msg)
        case _:
            raise ValueError(f"Invalid logging verbosity level: {log_scope}")
    return


def scale_lrs_by_world_size(
    lrs: dict[str, float],
    group: dist.ProcessGroup | None = None,
    inv_scale: bool = False,
    verbose_level: Literal["only-zero-rank", "world"] | None = None,
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
        verbose_level (Literal["only-zero-rank", "world"] | None): Verbosity level for logging scaled values.
            - "only-zero-rank": Log only from the main process (rank 0).
            - "world": Log from all processes in the distributed environment.
            -  None: No logging.

    Returns:
        dict[str, float]: The learning-rate configuration with scaled values.

    """
    world_size = dist.get_world_size(group=group)

    if inv_scale:
        scale = 1 / math.sqrt(world_size)
    else:
        scale = math.sqrt(world_size)

    for name, value in lrs.items():
        old_value = value
        new_value = value * scale
        if verbose_level is not None:
            log_dist(
                f"lr {name.upper()}: {new_value}; OLD: {old_value}",
                log_scope=verbose_level,
                group=group,
            )
        lrs[name] = new_value
    return lrs


def get_local_rank(group: dist.ProcessGroup | None = None) -> int:
    """Gets the local rank of the current process in a distributed setting."""
    if dist.is_initialized() and group is not None:
        return dist.get_rank(group=group)
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


def is_local_zero_rank() -> bool:
    """Checks if the current process is the main process (rank 0) for the local node in a distributed setting."""
    rank = get_local_rank()
    if rank != 0:
        return False
    return True
