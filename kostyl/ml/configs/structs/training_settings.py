import os
from typing import Literal
from typing import cast

import torch
from pydantic import Field
from pydantic import model_validator
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

from kostyl.utils.logging import setup_logger
from kostyl.utils.pydantic_models import StrictPydanticBaseModel


logger = setup_logger(fmt="only_message")

PRECISION = Literal[
    64,
    32,
    16,
    "transformer-engine",
    "transformer-engine-float16",
    "16-true",
    "16-mixed",
    "bf16-true",
    "bf16-mixed",
    "32-true",
    "64-true",
    "64",
    "32",
    "16",
    "bf16",
]
DTYPE = Literal["float32", "float16", "bfloat16", "float64"]


class SingleDeviceStrategyConfig(StrictPydanticBaseModel):
    """Single device strategy configuration."""

    kind: Literal["single-device"]


class FSDP1StrategyConfig(StrictPydanticBaseModel):
    """Fully Sharded Data Parallel (FSDP) strategy configuration."""

    kind: Literal["fsdp1"]
    param_dtype: DTYPE | None = None
    reduce_dtype: DTYPE | None = None
    buffer_dtype: DTYPE | None = None
    keep_low_precision_grads: bool = False
    use_cpu_offload: bool = False


class FSDP2StrategyConfig(StrictPydanticBaseModel):
    """Fully Sharded Data Parallel (FSDP) strategy configuration."""

    kind: Literal["fsdp2"]
    param_dtype: DTYPE | None = None
    reduce_dtype: DTYPE | None = None
    output_dtype: DTYPE | None = None
    use_cpu_offload: bool = False


_DDP_HOOKS = tuple(name for name in default_hooks.__all__ if name.endswith("_hook"))


class DDPStrategyConfig(StrictPydanticBaseModel):
    """Distributed Data Parallel (DDP) strategy configuration."""

    kind: Literal["ddp"]
    find_unused_parameters: bool = False
    hook: str | None = None

    @model_validator(mode="after")
    def _validate_hook(self) -> "DDPStrategyConfig":
        if self.hook is not None and self.hook not in _DDP_HOOKS:
            raise ValueError(
                f"Invalid DDP hook '{self.hook}'. Available hooks: {', '.join(_DDP_HOOKS)}"
            )
        return self


SupportedStrategies = (
    FSDP1StrategyConfig
    | FSDP2StrategyConfig
    | SingleDeviceStrategyConfig
    | DDPStrategyConfig
)


class MultiNodeDevices(StrictPydanticBaseModel):
    """Configuration for multi-node training devices."""

    node_rank: int
    node_devices: list[int] | int


def _as_device_list(devices: list[int] | int) -> list[int]:
    return list(range(devices)) if isinstance(devices, int) else devices


def _torchrun_node_rank() -> int:
    # torchrun exposes the node index as GROUP_RANK (it never sets NODE_RANK), and
    # Lightning's TorchElasticEnvironment reads `node_rank` from the same variable,
    # so the config and Lightning always agree on which node entry is local.
    return int(os.getenv("GROUP_RANK", "-1"))


def _check_node_layout(nodes: list[MultiNodeDevices]) -> None:
    ranks = sorted(node.node_rank for node in nodes)
    if ranks != list(range(len(nodes))):
        raise ValueError(
            f"Multi-node `node_rank` values must be unique and cover 0..{len(nodes) - 1}, got {ranks}."
        )
    # Lightning's TorchElasticEnvironment requires devices-per-node * num_nodes == WORLD_SIZE,
    # so every node has to run the same number of processes.
    device_counts = {
        node.node_rank: len(_as_device_list(node.node_devices)) for node in nodes
    }
    if len(set(device_counts.values())) > 1:
        raise ValueError(
            f"All nodes must use the same number of devices, got {device_counts} (node_rank -> device count)."
        )


class LightningTrainerParameters(StrictPydanticBaseModel):
    """Lightning Trainer parameters configuration."""

    accelerator: str
    max_epochs: int
    strategy: SupportedStrategies
    val_check_interval: int | float = 1.0
    check_val_every_n_epoch: int | None = 1
    devices: list[int] | list[MultiNodeDevices] | int
    precision: PRECISION
    log_every_n_steps: int = Field(default=50, ge=1)
    accumulate_grad_batches: int = Field(default=1, ge=1)
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    limit_predict_batches: int | float | None = None

    @property
    def is_multi_node(self) -> bool:
        """Returns True if the devices configuration indicates multi-node training."""
        return (
            isinstance(self.devices, list)
            and len(self.devices) > 0
            and all(isinstance(d, MultiNodeDevices) for d in self.devices)
        )

    @property
    def local_devices(self) -> list[int]:
        """Returns the list of devices for the current node."""
        if not self.is_multi_node:
            return _as_device_list(cast(list[int] | int, self.devices))

        node_rank = _torchrun_node_rank()
        for node in cast(list[MultiNodeDevices], self.devices):
            if node.node_rank == node_rank:
                return _as_device_list(node.node_devices)
        raise ValueError(
            f"No devices found for node rank {node_rank} (GROUP_RANK={os.getenv('GROUP_RANK')})."
        )

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the training configuration."""
        if self.is_multi_node:
            return len(cast(list[MultiNodeDevices], self.devices))
        return 1

    @model_validator(mode="after")
    def _validate_devices_and_accelerator(self) -> "LightningTrainerParameters":  # noqa: C901
        if self.is_multi_node:
            _check_node_layout(cast(list[MultiNodeDevices], self.devices))
            if not (
                torch.distributed.is_available()
                and torch.distributed.is_torchelastic_launched()
            ):
                raise ValueError(
                    "Multi-node training configuration must be launched with torchrun (TorchElastic). "
                    "For a single-node run, set `devices` to a plain list of device ids."
                )
            if int(os.getenv("GROUP_RANK", "-1")) < 0:
                raise ValueError(
                    "Multi-node training configuration requires GROUP_RANK environment variable to be set."
                )
            if int(os.getenv("GROUP_WORLD_SIZE", "-1")) != self.num_nodes:
                raise ValueError(
                    f"Mismatch between GROUP_WORLD_SIZE ({os.getenv('GROUP_WORLD_SIZE')}) and the number of nodes specified in devices ({self.num_nodes})."
                )

        match self.accelerator:
            case "cuda":
                if not torch.cuda.is_available():
                    raise ValueError(
                        "CUDA accelerator specified but no CUDA devices found."
                    )

                devices = self.local_devices
                if self.is_multi_node and int(
                    os.getenv("LOCAL_WORLD_SIZE", "-1")
                ) != len(devices):
                    raise ValueError(
                        f"Mismatch between LOCAL_WORLD_SIZE ({os.getenv('LOCAL_WORLD_SIZE')}) and the number of devices specified for the current node ({len(devices)})."
                    )

                available_devices = torch.cuda.device_count()
                if not all(0 <= d < available_devices for d in devices):
                    missing = [d for d in devices if not (0 <= d < available_devices)]
                    raise ValueError(
                        f"Requested CUDA devices {missing} are out of range. Available devices: {available_devices}."
                    )
            case "mps":
                if not torch.backends.mps.is_available():
                    raise ValueError(
                        "MPS accelerator specified but no MPS devices found."
                    )
            case "cpu":
                if not torch.cpu.is_available():
                    raise ValueError(
                        "CPU accelerator specified but no CPU devices found."
                    )
            case _:
                raise ValueError(f"Unsupported accelerator type: {self.accelerator}")
        return self


class EarlyStoppingConfig(StrictPydanticBaseModel):
    """Early stopping configuration."""

    monitor: str
    mode: str
    patience: int = Field(default=5, ge=1)
    min_delta: float = Field(default=0.01, ge=0)


class CheckpointConfig(StrictPydanticBaseModel):
    """Model checkpointing configuration."""

    save_top_k: int = 4
    monitor: str = "val_loss"
    mode: str = "min"
    filename: str = "{epoch:02d}-{val_loss:.2f}"
    save_weights_only: bool = True


class DataConfig(StrictPydanticBaseModel):
    """Data configuration."""

    datasets: dict[str, str]
    batch_size: int
    num_workers: int = Field(ge=1)
    data_columns: list[str]


class TrainingSettings(StrictPydanticBaseModel):
    """Training parameters configuration."""

    trainer: LightningTrainerParameters
    early_stopping: EarlyStoppingConfig | None = None
    checkpoint: CheckpointConfig = CheckpointConfig()
    data: DataConfig
