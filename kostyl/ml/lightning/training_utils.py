from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Literal
from typing import cast

import lightning as L
import torch
import torch.distributed as dist
from clearml import OutputModel
from clearml import Task
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import MixedPrecision
from torch.nn import Module

from kostyl.ml.configs import CheckpointConfig
from kostyl.ml.configs import DDPStrategyConfig
from kostyl.ml.configs import EarlyStoppingConfig
from kostyl.ml.configs import FSDP1StrategyConfig
from kostyl.ml.configs import SingleDeviceStrategyConfig
from kostyl.ml.lightning.callbacks import setup_checkpoint_callback
from kostyl.ml.lightning.callbacks import setup_early_stopping_callback
from kostyl.ml.lightning.loggers import setup_tb_logger
from kostyl.ml.registry_uploader import ClearMLRegistryUploaderCallback
from kostyl.utils.logging import setup_logger


TRAINING_STRATEGIES = (
    FSDP1StrategyConfig | DDPStrategyConfig | SingleDeviceStrategyConfig
)

logger = setup_logger(add_rank=True)


def estimate_total_steps(
    trainer: L.Trainer, process_group: ProcessGroup | None = None
) -> int:
    """
    Estimates the total number of training steps based on the
    dataloader length, accumulation steps, and distributed world size.
    """  # noqa: D205
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


@dataclass
class Callbacks:
    """Dataclass to hold PyTorch Lightning callbacks."""

    checkpoint: ModelCheckpoint
    lr_monitor: LearningRateMonitor
    early_stopping: EarlyStopping | None = None

    def to_list(self) -> list[Callback]:
        """Convert dataclass fields to a list of Callbacks. None values are omitted."""
        callbacks: list[Callback] = [
            getattr(self, field.name)
            for field in fields(self)
            if getattr(self, field.name) is not None
        ]
        return callbacks


def setup_callbacks(
    task: Task,
    root_path: Path,
    checkpoint_cfg: CheckpointConfig,
    early_stopping_cfg: EarlyStoppingConfig | None,
    output_model: OutputModel,
    checkpoint_upload_strategy: Literal["only-best", "every-checkpoint"],
    config_dict: dict[str, str] | None = None,
    enable_tag_versioning: bool = False,
) -> Callbacks:
    """
    Set up PyTorch Lightning callbacks for training.

    Creates and configures a set of callbacks including checkpoint saving,
    learning rate monitoring, model registry uploading, and optional early stopping.

    Args:
        task: ClearML task for organizing checkpoints by task name and ID.
        root_path: Root directory for saving checkpoints.
        checkpoint_cfg: Configuration for checkpoint saving behavior.
        checkpoint_upload_strategy: Model upload strategy:
            - `"only-best"`: Upload only the best checkpoint based on monitored metric.
            - `"every-checkpoint"`: Upload every saved checkpoint.
        output_model: ClearML OutputModel instance for model registry integration.
        early_stopping_cfg: Configuration for early stopping. If None, early stopping
            is disabled.
        config_dict: Optional configuration dictionary to store with the model
            in the registry.
        enable_tag_versioning: Whether to auto-increment version tags (e.g., "v1.0")
            on the uploaded model.

    Returns:
        Callbacks dataclass containing configured checkpoint, lr_monitor,
        and optionally early_stopping callbacks.

    """
    lr_monitor = LearningRateMonitor(
        logging_interval="step", log_weight_decay=True, log_momentum=False
    )
    model_uploader = ClearMLRegistryUploaderCallback(
        output_model=output_model,
        config_dict=config_dict,
        verbose=True,
        enable_tag_versioning=enable_tag_versioning,
    )
    checkpoint_callback = setup_checkpoint_callback(
        root_path / "checkpoints" / task.name / task.id,
        checkpoint_cfg,
        registry_uploader_callback=model_uploader,
        uploading_strategy=checkpoint_upload_strategy,
    )
    if early_stopping_cfg is not None:
        early_stopping_callback = setup_early_stopping_callback(early_stopping_cfg)
    else:
        early_stopping_callback = None

    callbacks = Callbacks(
        checkpoint=checkpoint_callback,
        lr_monitor=lr_monitor,
        early_stopping=early_stopping_callback,
    )
    return callbacks


def setup_loggers(task: Task, root_path: Path) -> list[TensorBoardLogger]:
    """
    Set up PyTorch Lightning loggers for training.

    Args:
        task: ClearML task used to organize log directories by task name and ID.
        root_path: Root directory for storing TensorBoard logs.

    Returns:
        List of configured TensorBoard loggers.

    """
    loggers = [
        setup_tb_logger(root_path / "runs" / task.name / task.id),
    ]
    return loggers


def setup_strategy(
    strategy_settings: TRAINING_STRATEGIES,
    devices: list[int] | int,
    auto_wrap_policy: set[type[Module]] | None = None,
) -> Literal["auto"] | FSDPStrategy | DDPStrategy:
    """
    Configure and return a PyTorch Lightning training strategy.

    Args:
        strategy_settings: Strategy configuration object. Must be one of:
            - `FSDP1StrategyConfig`: Fully Sharded Data Parallel strategy (requires 2+ devices).
            - `DDPStrategyConfig`: Distributed Data Parallel strategy (requires 2+ devices).
            - `SingleDeviceStrategyConfig`: Single device training (requires exactly 1 device).
        devices: Device(s) to use for training. Either a list of device IDs or
            a single integer representing the number of devices.
        auto_wrap_policy: Set of module types that should be wrapped for FSDP.
            Required when using `FSDP1StrategyConfig`, ignored otherwise.

    Returns:
        Configured strategy: `FSDPStrategy`, `DDPStrategy`, or `"auto"` for single device.

    Raises:
        ValueError: If device count doesn't match strategy requirements or
            if `auto_wrap_policy` is missing for FSDP.

    """
    if isinstance(devices, list):
        num_devices = len(devices)
    else:
        num_devices = devices

    match strategy_settings:
        case FSDP1StrategyConfig():
            if num_devices < 2:
                raise ValueError("FSDP strategy requires multiple devices.")

            if auto_wrap_policy is None:
                raise ValueError("auto_wrap_policy must be provided for FSDP strategy.")

            mixed_precision_config = MixedPrecision(
                param_dtype=getattr(torch, strategy_settings.param_dtype),
                reduce_dtype=getattr(torch, strategy_settings.reduce_dtype),
                buffer_dtype=getattr(torch, strategy_settings.buffer_dtype),
            )
            strategy = FSDPStrategy(
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision_config,
            )
        case DDPStrategyConfig():
            if num_devices < 2:
                raise ValueError("DDP strategy requires at least two devices.")
            strategy = DDPStrategy(
                find_unused_parameters=strategy_settings.find_unused_parameters
            )
        case SingleDeviceStrategyConfig():
            if num_devices != 1:
                raise ValueError("SingleDevice strategy requires exactly one device.")
            strategy = "auto"
        case _:
            raise ValueError(
                f"Unsupported strategy type: {type(strategy_settings.trainer.strategy)}"
            )
    return strategy
