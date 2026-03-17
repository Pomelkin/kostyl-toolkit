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
        dp_world_size = dist.get_world_size(dp_process_group)
    else:
        dp_world_size = 1

    datamodule = trainer.datamodule  # type: ignore
    if datamodule is None:
        raise ValueError("Trainer must have a datamodule to estimate total steps.")
    datamodule = cast(L.LightningDataModule, datamodule)

    logger.info("Loading `train_dataloader` to estimate number of stepping batches.")
    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()

    # Estimate number of batches per epoch
    try:
        raw_len = len(train_dataloader)
        dl_len = raw_len // dp_world_size  # Assuming the distributed sampler
    except TypeError as e:
        # IterableDataset without __len__
        if isinstance(trainer.limit_train_batches, int):
            dl_len = trainer.limit_train_batches
            raw_len = None
        else:
            raise ValueError(
                "Cannot estimate steps for IterableDataset without __len__ unless limit_train_batches is an int."
            ) from e

    limit_batches = trainer.limit_train_batches

    if limit_batches is None or limit_batches == 1.0:
        effective_len = dl_len
    elif isinstance(limit_batches, int):
        effective_len = min(dl_len, limit_batches)
    elif isinstance(limit_batches, float):
        effective_len = int(dl_len * limit_batches)
    else:
        raise RuntimeError(
            f"Unexpected type for trainer.limit_train_batches: {type(limit_batches)}"
        )

    if trainer.max_epochs is None:
        raise ValueError("Trainer must have `max_epochs` set to estimate total steps.")

    steps_per_epoch = effective_len // trainer.accumulate_grad_batches
    total_steps = steps_per_epoch * trainer.max_epochs

    logger.info(
        f"Total optimization steps: {total_steps} (Batches per epoch (per GPU): {steps_per_epoch})\n"
        f"  Details:\n"
        f"  -> Raw Dataloader len: {raw_len}\n"
        f"  -> Global Batch Size factor (World Size): {dp_world_size}\n"
        f"  -> Effective len (after limits/overfit): {effective_len}\n"
        f"  -> Accumulate grad batches: {trainer.accumulate_grad_batches}\n"
        f"  -> Max Epochs: {trainer.max_epochs}"
    )
    return total_steps
