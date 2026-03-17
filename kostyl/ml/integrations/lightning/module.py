from collections import defaultdict
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any
from typing import cast
from typing import override

import lightning as L
import torch
from lightning.pytorch.strategies import FSDPStrategy
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchmetrics import Metric
from torchmetrics import MetricCollection
from transformers import PretrainedConfig
from transformers import PreTrainedModel

from kostyl.ml.integrations.lightning.metrics_formatting import apply_suffix
from kostyl.ml.optim.schedulers import BaseScheduler
from kostyl.utils import is_overridden
from kostyl.utils import setup_logger


module_logger = setup_logger(fmt="only_message")


class KostylLightningModule(L.LightningModule):
    """
    Custom PyTorch Lightning Module.

    Child classes must implement this methods:
    - `model_instance`: to return the underlying model instance (e.g., a Hugging Face PreTrainedModel).
    - `model_config`: to return the model configuration if available.
    - `grad_clip_val`: to return the gradient clipping value from hyperparameters if set.
    """

    @property
    def model_instance(self) -> PreTrainedModel | nn.Module:
        """Returns the underlying model."""
        raise NotImplementedError

    @property
    def model_config(self) -> PretrainedConfig | None:
        """Returns the model configuration if available."""
        raise NotImplementedError

    def get_configuration_dict(self) -> dict[str, Any]:
        """Returns a dictionary of configuration values to be saved into the checkpoint."""
        model = self.model_instance
        cfg = defaultdict(dict)

        if is_overridden(self.__class__, "model_config"):
            cfg_instance = self.model_config
        elif hasattr(model, "config"):
            cfg_instance = model.config
        else:
            raise RuntimeError(
                "Cannot find model configuration. Please override `model_config` property or ensure the model has a `config` attribute."
            )

        if hasattr(cfg_instance, "to_dict"):
            cfg["config"] = cfg_instance.to_dict()  # type: ignore

        hf_peft_config_loaded = getattr(self, "_hf_peft_config_loaded", False)
        if hf_peft_config_loaded:
            model = cast(PreTrainedModel, model)
            active_adapters = model.active_adapters()
            for adapter_name in active_adapters:
                curr_adapter_cfg = self.peft_config[adapter_name]  # type: ignore
                cfg["peft_config"][adapter_name] = curr_adapter_cfg.to_dict()  # type: ignore
        return cfg

    @property
    def grad_clip_val(self) -> float | None:
        """Returns the gradient clipping value from hyperparameters if set."""
        raise NotImplementedError

    @override
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        cfg = self.get_configuration_dict()
        checkpoint["kostyl_config"].update(cfg)
        return

    @override
    def on_before_optimizer_step(self, optimizer) -> None:
        grad_clip_val = self.grad_clip_val
        if grad_clip_val is None:
            return

        if not isinstance(self.trainer.strategy, FSDPStrategy):
            norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=grad_clip_val
            )
        else:
            module: FSDP = self.trainer.strategy.model  # type: ignore
            norm = module.clip_grad_norm_(max_norm=grad_clip_val)  # type: ignore
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
        scheduler_state_dict = apply_suffix(scheduler_state_dict, "scheduler")
        self.log_dict(
            scheduler_state_dict,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        return
