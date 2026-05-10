from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from kostyl.utils.logging import setup_logger


logger = setup_logger(fmt="only_message")


class AdamConfig(BaseModel):
    """Adam optimizer hyperparameters configuration."""

    type: Literal["AdamW", "Adam"] = "AdamW"
    betas: tuple[float, float] = (0.9, 0.999)


class MuonConfig(BaseModel):
    """Muon optimizer hyperparameters configuration."""

    type: Literal["Muon"]
    momentum: float = 0.95
    nesterov: bool = True
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
    ns_steps: int = 5


class AdamWithPrecisionConfig(BaseModel):
    """Adam optimizer with low-precision hyperparameters configuration."""

    type: Literal[
        "Adam8bit", "Adam4bit", "AdamFp8", "AdamW8bit", "AdamW4bit", "AdamWFp8"
    ]
    betas: tuple[float, float] = (0.9, 0.999)
    block_size: int
    bf16_stochastic_round: bool = False


OPTIMIZER_CONFIG = AdamConfig | AdamWithPrecisionConfig | MuonConfig
SCHEDULER = Literal[
    "linear",
    "cosine",
    "plateau-with-cosine-annealing",
    "plateau-with-linear-annealing",
]


class ScheduledParamConfig(BaseModel):
    """Base configuration for a scheduled hyperparameter."""

    scheduler_type: SCHEDULER | None = None

    freeze_ratio: float | None = Field(default=None, ge=0, le=1)
    warmup_ratio: float | None = Field(default=None, gt=0, lt=1, validate_default=False)
    warmup_value: float | None = Field(default=None, gt=0, validate_default=False)
    base_value: float
    final_value: float | None = Field(default=None, gt=0, validate_default=False)
    plateau_ratio: float | None = Field(
        default=None, gt=0, lt=1, validate_default=False
    )

    @model_validator(mode="after")
    def _validate_freeze_ratio(self) -> "ScheduledParamConfig":
        if self.scheduler_type is None and self.freeze_ratio is not None:
            logger.warning("use_scheduler is False, freeze_ratio will be ignored.")
            self.freeze_ratio = None
        return self

    @model_validator(mode="after")
    def _validate_warmup(self) -> "ScheduledParamConfig":
        if ((self.warmup_value is not None) or (self.warmup_ratio is not None)) and self.scheduler_type is None:  # fmt: skip
            logger.warning(
                "scheduler_type is None, warmup_value and warmup_ratio will be ignored."
            )
            self.warmup_value = None
            self.warmup_ratio = None
        if (self.warmup_value is None) != (self.warmup_ratio is None):  # fmt: skip
            raise ValueError(
                "Both warmup_value and warmup_ratio must be provided or neither"
            )
        return self

    @model_validator(mode="after")
    def _validate_final_value(self) -> "ScheduledParamConfig":
        if (self.scheduler_type in {"linear"}) and (self.final_value is not None):
            raise ValueError("If scheduler_type is 'linear', final_value must be None.")
        if (self.scheduler_type is None) and (self.final_value is not None):
            logger.warning("use_scheduler is False, final_value will be ignored.")
            self.final_value = None
        return self

    @model_validator(mode="after")
    def _validate_plateau_ratio(self) -> "ScheduledParamConfig":
        if self.scheduler_type is not None:
            if self.scheduler_type.startswith("plateau") and self.plateau_ratio is None:
                raise ValueError(
                    "If scheduler_type is 'plateau-with-*', plateau_ratio must be provided."
                )
            if (
                not self.scheduler_type.startswith("plateau")
                and self.plateau_ratio is not None
            ):
                logger.warning(
                    "scheduler_type is not 'plateau-with-*', plateau_ratio will be ignored."
                )
                self.plateau_ratio = None
        return self


class Lr(ScheduledParamConfig):
    """Learning rate hyperparameters configuration."""


class WeightDecay(ScheduledParamConfig):
    """Weight decay hyperparameters configuration."""


class HyperparamsConfig(BaseModel):
    """Model training hyperparameters configuration."""

    grad_clip_val: float | None = Field(default=None, gt=0, validate_default=False)
    optimizer: OPTIMIZER_CONFIG
    lr: Lr
    weight_decay: WeightDecay
