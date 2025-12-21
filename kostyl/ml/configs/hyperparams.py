from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from kostyl.utils.logging import setup_logger


logger = setup_logger(fmt="only_message")


class Optimizer(BaseModel):
    """Optimizer hyperparameters configuration."""

    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999


class Lr(BaseModel):
    """Learning rate hyperparameters configuration."""

    use_scheduler: bool = False
    warmup_iters_ratio: float | None = Field(
        default=None, gt=0, lt=1, validate_default=False
    )
    warmup_value: float | None = Field(default=None, gt=0, validate_default=False)
    base_value: float
    final_value: float | None = Field(default=None, gt=0, validate_default=False)

    @model_validator(mode="after")
    def validate_warmup(self) -> "Lr":
        """Validates the warmup parameters based on use_scheduler."""
        if (self.warmup_value is None) != (self.warmup_iters_ratio is None):  # fmt: skip
            raise ValueError(
                "Both warmup_value and warmup_iters_ratio must be provided or neither"
            )
        if ((self.warmup_value is not None) or (self.warmup_iters_ratio is not None)) and not self.use_scheduler:  # fmt: skip
            logger.warning(
                "use_scheduler is False, warmup_value and warmup_iters_ratio will be ignored."
            )
            self.warmup_value = None
            self.warmup_iters_ratio = None
        return self

    @model_validator(mode="after")
    def validate_final_value(self) -> "Lr":
        """Validates the final_value based on use_scheduler."""
        if self.use_scheduler and (self.final_value is None):
            raise ValueError("If use_scheduler is True, final_value must be provided.")
        if (not self.use_scheduler) and (self.final_value is not None):
            logger.warning("use_scheduler is False, final_value will be ignored.")
            self.final_value = None
        return self


class WeightDecay(BaseModel):
    """Weight decay hyperparameters configuration."""

    use_scheduler: bool = False
    base_value: float
    final_value: float | None = None

    @model_validator(mode="after")
    def validate_final_value(self) -> "WeightDecay":
        """Validates the final_value based on use_scheduler."""
        if self.use_scheduler and self.final_value is None:
            raise ValueError("If use_scheduler is True, final_value must be provided.")
        if not self.use_scheduler and self.final_value is not None:
            logger.warning("use_scheduler is False, final_value will be ignored.")
        return self


class HyperparamsConfig(BaseModel):
    """Model training hyperparameters configuration."""

    grad_clip_val: float | None = Field(default=None, gt=0, validate_default=False)
    optimizer: Optimizer = Optimizer()
    lr: Lr
    weight_decay: WeightDecay
