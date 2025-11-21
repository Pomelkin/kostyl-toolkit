import math
import os

import torch.distributed as dist

from kostyl.ml_core.configs import Lr
from kostyl.utils.logging import setup_logger


logger = setup_logger(add_rank=True)


def scale_lrs_by_world_size[Tlr: Lr](
    lr_config: Tlr,
    group: dist.ProcessGroup | None = None,
    config_name: str = "",
    inv_scale: bool = False,
) -> Tlr:
    """
    Scale learning-rate configuration values to match the active distributed world size.

    Args:
        lr_config (Lr): Learning-rate configuration whose values will be scaled.
        group (dist.ProcessGroup | None): Optional process group used to determine
            the target world size. Defaults to the global process group.
        config_name (str): Human-readable identifier included in log messages.
        inv_scale (bool): If True, use the inverse square-root scale factor.

    Returns:
        Tlr: The learning-rate configuration with scaled values.

    """
    world_size = dist.get_world_size(group=group)

    if inv_scale:
        scale = 1 / math.sqrt(world_size)
    else:
        scale = math.sqrt(world_size)

    logger.info(f"Scaling learning rates for world size: {world_size}")
    logger.info(f"Scale factor: {scale:.4f}")
    old_base = lr_config.base_value
    lr_config.base_value *= scale
    logger.info(f"New {config_name} lr BASE: {lr_config.base_value}; OLD: {old_base}")

    if lr_config.final_value is not None:
        old_final_value = lr_config.final_value
        lr_config.final_value *= scale
        logger.info(
            f"New {config_name} lr FINAL: {lr_config.final_value}; OLD: {old_final_value}"
        )

    if lr_config.warmup_value is not None:
        old_warmup_value = lr_config.warmup_value
        lr_config.warmup_value *= scale
        logger.info(
            f"New {config_name} lr WARMUP: {lr_config.warmup_value}; OLD: {old_warmup_value}"
        )
    return lr_config


def _get_rank() -> int:
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = int(os.environ.get("RANK", 0))
    return rank


def is_main_process() -> bool:
    """Checks if the current process is the main process (rank 0) in a distributed setting."""
    return _get_rank() == 0
