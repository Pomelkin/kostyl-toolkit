import math
from typing import Literal
from typing import cast

import lightning as L
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from kostyl.utils.logging import setup_logger


logger = setup_logger()


def estimate_total_steps(  # noqa: C901
    trainer: L.Trainer,
    dp_process_group: ProcessGroup | None = None,
    verbosity_level: Literal["rank-zero-only", "world"] = "world",
) -> int:
    """
    Estimates the total number of training steps with respect to data parallelism and gradient accumulation.

    Args:
        trainer: The PyTorch Lightning Trainer instance.
        dp_process_group: The data parallel process group. If None, the world process group will be used.
        verbosity_level: Verbosity level for logging the estimated steps.
            - "rank-zero-only": Log only from the main process (rank 0).
            - "world": Log from all processes in the distributed environment.

    """
    if dist.is_initialized():
        dp_world_size = dist.get_world_size(dp_process_group)
    else:
        dp_world_size = 1

    datamodule = trainer.datamodule  # type: ignore
    if datamodule is None:
        raise ValueError("Trainer must have a datamodule to estimate total steps.")
    datamodule = cast(L.LightningDataModule, datamodule)

    datamodule.setup("fit")

    train_dataloader: DataLoader = datamodule.train_dataloader()

    # Estimate number of batches per epoch
    try:
        raw_len = len(train_dataloader)
        if dist.is_initialized() and not isinstance(
            train_dataloader.sampler, DistributedSampler
        ):
            dl_len = raw_len // dp_world_size  # Assuming the distributed sampler
        else:
            dl_len = raw_len
            # Adjust for global batch size
            # For non-distributed env dp_world_size = 1, so this is a no-op in that case
            raw_len = raw_len * dp_world_size
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

    steps_per_epoch = math.ceil(effective_len / trainer.accumulate_grad_batches)
    total_steps = steps_per_epoch * max(trainer.max_epochs, 1)

    msg = (
        f"Total optimization steps: {total_steps} (Batches per epoch (per GPU): {steps_per_epoch})\n"
        "  Details:\n"
        f"  -> Raw Dataloader len: {raw_len}\n"
        f"  -> Global Batch Size factor (DP World Size): {dp_world_size}\n"
        f"  -> Effective len (after limits/overfit/distributed factor): {effective_len}\n"
        f"  -> Accumulate grad batches: {trainer.accumulate_grad_batches}\n"
        f"  -> Max Epochs: {trainer.max_epochs}"
    )

    if verbosity_level == "rank-zero-only":
        logger.log_rank_zero(
            level="INFO",
            msg=msg,
        )
    elif verbosity_level == "world":
        logger.info(msg)
    return total_steps
