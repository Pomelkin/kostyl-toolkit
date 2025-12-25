from collections.abc import Callable
from collections.abc import Mapping
from typing import Any
from typing import override

import lightning as L
import torch
import torch.distributed as dist
from lightning.pytorch.strategies import FSDPStrategy
from torch import nn
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchmetrics import Metric
from torchmetrics import MetricCollection
from transformers import PretrainedConfig
from transformers import PreTrainedModel

from kostyl.ml.metrics_formatting import apply_suffix
from kostyl.ml.schedulers.base import BaseScheduler
from kostyl.utils import setup_logger


module_logger = setup_logger(fmt="only_message")


class KostylLightningModule(L.LightningModule):
    """Custom PyTorch Lightning Module with logging, checkpointing, and distributed training utilities."""

    @property
    def process_group(self) -> ProcessGroup | None:
        """Returns the data parallel process group for distributed training."""
        return self.get_process_group()

    def get_process_group(self) -> ProcessGroup | None:
        """
        Retrieves the data parallel process group for distributed training.

        This method checks if distributed processing is initialized. If a device mesh is provided,
        it extracts the data parallel mesh and returns its process group, unless the mesh size is 1,
        in which case it logs a warning and returns None. If no device mesh is provided, it returns
        the world process group.

        Returns:
            ProcessGroup | None: The data parallel process group if available and valid, otherwise None.

        """
        if not dist.is_initialized():
            return None

        if self.device_mesh is not None:
            dp_mesh = self.device_mesh["data_parallel"]
            if dp_mesh.size() == 1:
                module_logger.warning("Data parallel mesh size is 1, returning None")
                return None
            dp_pg = dp_mesh.get_group()
        else:
            dp_pg = dist.group.WORLD
        return dp_pg

    @property
    def model_instance(self) -> PreTrainedModel | nn.Module:
        """Returns the underlying model."""
        raise NotImplementedError

    @property
    def model_config(self) -> PretrainedConfig | None:
        """Returns the model configuration if available."""
        model = self.model_instance
        if hasattr(model, "config"):
            return model.config  # type: ignore
        return None

    @property
    def grad_clip_val(self) -> float | None:
        """Returns the gradient clipping value from hyperparameters if set."""
        raise NotImplementedError

    @override
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        model = self.model_instance
        if hasattr(model, "config"):
            cfg = model.config
            if hasattr(cfg, "to_dict"):
                checkpoint["config"] = cfg.to_dict()  # type: ignore

        if hasattr(model, "peft_config"):
            peft_cfg = model.peft_config
            if isinstance(peft_cfg, dict):
                checkpoint["peft_config"] = {
                    k: v.to_dict()
                    for k, v in peft_cfg.items()  # type: ignore
                }
            else:
                checkpoint["peft_config"] = {"default": peft_cfg.to_dict()}  # type: ignore
        return

    @override
    def on_before_optimizer_step(self, optimizer) -> None:
        grad_clip_val = self.grad_clip_val
        if grad_clip_val is None:
            return

        if not isinstance(self.trainer.strategy, FSDPStrategy):
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_val)
        else:
            module: FSDP = self.trainer.strategy.model  # type: ignore
            norm = module.clip_grad_norm_(grad_clip_val)
        self.log(
            "grad_norm",
            norm,
            logger=True,
            sync_dist=False,
            on_step=True,
            on_epoch=False,
        )
        return

    @override
    def log_dict(
        self,
        dictionary: Mapping[str, Metric | torch.Tensor | int | float]
        | MetricCollection,
        prog_bar: bool = False,
        logger: bool | None = None,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        reduce_fx: str | Callable[..., Any] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Any | None = None,
        add_dataloader_idx: bool = True,
        batch_size: int | None = None,
        rank_zero_only: bool = False,
        stage: str | None = None,
    ) -> None:
        if stage is not None:
            if not isinstance(dictionary, MetricCollection):
                dictionary = apply_suffix(
                    metrics=dictionary,
                    suffix=stage,
                    add_dist_rank=False,
                )
            else:
                module_logger.warning_once(
                    "Stage suffixing for MetricCollection is not implemented. Skipping suffixing."
                )
        super().log_dict(
            dictionary,
            prog_bar,
            logger,
            on_step,
            on_epoch,
            reduce_fx,
            enable_graph,
            sync_dist,
            sync_dist_group,
            add_dataloader_idx,
            batch_size,
            rank_zero_only,
        )
        return

    def log_scheduled_values(self) -> None:
        """
        Logs the current values from the learning rate scheduler.

        This method retrieves the current value from the scheduler, applies a suffix to the keys,
        and logs the resulting dictionary using the logger. The logging is performed on each step
        but not on each epoch, without displaying on the progress bar or synchronizing across
        distributed processes.
        """
        scheduler: BaseScheduler = self.lr_schedulers()  # type: ignore
        if not isinstance(scheduler, BaseScheduler):
            module_logger.warning_once(
                "Scheduler is not an instance of BaseScheduler. Skipping scheduled values logging."
            )
            return
        scheduler_state_dict = scheduler.current_value()
        scheduler_state_dict = apply_suffix(scheduler_state_dict, "scheduled")
        self.log_dict(
            scheduler_state_dict,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        return
