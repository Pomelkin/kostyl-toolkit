import math
import os
from typing import Literal

import torch.distributed as dist

from kostyl.utils.logging import setup_logger


logger = setup_logger(add_rank=True)


def log_dist(msg: str, how: Literal["only-zero-rank", "world"]) -> None:
    """
    Log a message in a distributed environment based on the specified verbosity level.

    Args:
        msg (str): The message to log.
        how (Literal["only-zero-rank", "world"]): The verbosity level for logging.
            - "only-zero-rank": Log only from the main process (rank 0).
            - "world": Log from all processes in the distributed environment.

    """
    match how:
        case _ if not dist.is_initialized():
            logger.warning_once(
                "Distributed logging requested but torch.distributed is not initialized."
            )
            logger.info(msg)
        case "only-zero-rank":
            if is_main_process():
                logger.info(msg)
        case "world":
            logger.info(msg)
        case _:
            logger.warning_once(
                f"Invalid logging verbosity level requested: {how}. Message not logged."
            )
    return


def scale_lrs_by_world_size(
    lrs: dict[str, float],
    group: dist.ProcessGroup | None = None,
    config_name: str = "",
    inv_scale: bool = False,
    verbose: Literal["only-zero-rank", "world"] | None = None,
) -> dict[str, float]:
    """
    Scale learning-rate configuration values to match the active distributed world size.

    Note:
        The value in the `lrs` will be modified in place.

    Args:
        lrs (dict[str, float]): A dictionary of learning rate names and their corresponding values to be scaled.
        group (dist.ProcessGroup | None): Optional process group used to determine
            the target world size. Defaults to the global process group.
        config_name (str): Human-readable identifier included in log messages.
        inv_scale (bool): If True, use the inverse square-root scale factor.
        verbose (Literal["only-zero-rank", "world"] | None): Verbosity level for logging scaled values.
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
        if verbose is not None:
            log_dist(
                f"New {config_name} lr {name.upper()}: {new_value}; OLD: {old_value}",
                verbose,
            )
        lrs[name] = new_value
    return lrs


def get_rank() -> int:
    """Gets the rank of the current process in a distributed setting."""
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


def is_main_process() -> bool:
    """Checks if the current process is the main process (rank 0) in a distributed setting."""
    rank = get_rank()
    if rank != 0:
        return False
    return True
