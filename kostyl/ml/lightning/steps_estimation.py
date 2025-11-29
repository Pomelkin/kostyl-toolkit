from typing import cast

import lightning as L
import torch.distributed as dist
from torch.distributed import ProcessGroup

from kostyl.utils.logging import setup_logger


logger = setup_logger(add_rank=True)


def estimate_total_steps(
    trainer: L.Trainer, process_group: ProcessGroup | None = None
) -> int:
    """Estimates the total number of training steps for a given PyTorch Lightning Trainer."""
    if dist.is_initialized():
        world_size = dist.get_world_size(process_group)
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
        f"Total steps: {total_steps} (per-epoch: {steps_per_epoch})\n"
        f"-> Dataloader len: {dataloader_len}\n"
        f"-> Accumulate grad batches: {trainer.accumulate_grad_batches}\n"
        f"-> Epochs: {trainer.max_epochs}\n "
        f"-> World size: {world_size}"
    )
    return total_steps
