from typing import cast

import lightning as L
import torch.distributed as dist
from torch.distributed import ProcessGroup

from kostyl.utils.logging import setup_logger


logger = setup_logger()


def estimate_total_steps(
    trainer: L.Trainer, dp_process_group: ProcessGroup | None = None
) -> int:
    """
    Estimates the total number of training steps with respect to data parallelism and gradient accumulation.

    Args:
        trainer: The PyTorch Lightning Trainer instance.
        dp_process_group: The data parallel process group. If None, the world process group will be used.

    """
    if dist.is_initialized():
        world_size = dist.get_world_size(dp_process_group)
    else:
        world_size = 1

    datamodule = trainer.datamodule  # type: ignore
    if datamodule is None:
        raise ValueError("Trainer must have a datamodule to estimate total steps.")
    datamodule = cast(L.LightningDataModule, datamodule)

    logger.info("Loading `train_dataloader` to estimate number of stepping batches.")
    datamodule.setup("fit")

    dataloader_len = len(datamodule.train_dataloader())
    steps_per_epoch = dataloader_len // trainer.accumulate_grad_batches // world_size

    if trainer.max_epochs is None:
        raise ValueError("Trainer must have `max_epochs` set to estimate total steps.")
    total_steps = steps_per_epoch * trainer.max_epochs

    logger.info(
        f"Total steps: {total_steps} (per-epoch: {steps_per_epoch}) "
        f"-> Dataloader len: {dataloader_len} "
        f"-> Accumulate grad batches: {trainer.accumulate_grad_batches} "
        f"-> Epochs: {trainer.max_epochs} "
        f"-> DataParallel size: {world_size}"
    )
    return total_steps
