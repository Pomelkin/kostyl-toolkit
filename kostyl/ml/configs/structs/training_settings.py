from typing import Literal

import torch
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from kostyl.utils.logging import setup_logger


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


class SingleDeviceStrategyConfig(BaseModel):
    """Single device strategy configuration."""

    type: Literal["single_device"]


class FSDP1StrategyConfig(BaseModel):
    """Fully Sharded Data Parallel (FSDP) strategy configuration."""

    type: Literal["fsdp1"]
    param_dtype: DTYPE | None = None
    reduce_dtype: DTYPE | None = None
    buffer_dtype: DTYPE | None = None
    keep_low_precision_grads: bool = False
    use_cpu_offload: bool = False


class FSDP2StrategyConfig(BaseModel):
    """Fully Sharded Data Parallel (FSDP) strategy configuration."""

    type: Literal["fsdp2"]
    param_dtype: DTYPE | None = None
    reduce_dtype: DTYPE | None = None
    output_dtype: DTYPE | None = None
    use_cpu_offload: bool = False


class DDPStrategyConfig(BaseModel):
    """Distributed Data Parallel (DDP) strategy configuration."""

    type: Literal["ddp"]
    find_unused_parameters: bool = False


SUPPORTED_STRATEGIES = (
    FSDP1StrategyConfig
    | FSDP2StrategyConfig
    | SingleDeviceStrategyConfig
    | DDPStrategyConfig
)


class LightningTrainerParameters(BaseModel):
    """Lightning Trainer parameters configuration."""

    accelerator: str
    max_epochs: int
    strategy: SUPPORTED_STRATEGIES
    val_check_interval: int | float
    devices: list[int] | int
    precision: PRECISION
    log_every_n_steps: int = Field(default=50, ge=1)
    accumulate_grad_batches: int = Field(default=1, ge=1)
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    limit_predict_batches: int | float | None = None

    @model_validator(mode="after")
    def _validate_devices_and_accelerator(self) -> "LightningTrainerParameters":
        match self.accelerator:
            case "cuda":
                if not torch.cuda.is_available():
                    raise ValueError(
                        "CUDA accelerator specified but no CUDA devices found."
                    )

                devices = (
                    self.devices
                    if isinstance(self.devices, list)
                    else list(range(self.devices))
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


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""

    monitor: str
    mode: str
    patience: int = Field(default=5, ge=1)
    min_delta: float = Field(default=0.01, ge=0)


class CheckpointConfig(BaseModel):
    """Model checkpointing configuration."""

    save_top_k: int = 4
    monitor: str = "val_loss"
    mode: str = "min"
    filename: str = "{epoch:02d}-{val_loss:.2f}"
    save_weights_only: bool = True


class DataConfig(BaseModel):
    """Data configuration."""

    datasets: dict[str, str]
    batch_size: int
    num_workers: int = Field(ge=1)
    data_columns: list[str]


class TrainingSettings(BaseModel):
    """Training parameters configuration."""

    trainer: LightningTrainerParameters
    early_stopping: EarlyStoppingConfig | None = None
    checkpoint: CheckpointConfig = CheckpointConfig()
    data: DataConfig
