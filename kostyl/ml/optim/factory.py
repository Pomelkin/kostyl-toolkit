from typing import TypedDict
from typing import Unpack

from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT

from kostyl.ml.configs import OPTIMIZER_CONFIG
from kostyl.ml.configs import SCHEDULER
from kostyl.ml.configs import AdamConfig
from kostyl.ml.configs import AdamWithPrecisionConfig
from kostyl.ml.configs import MuonConfig
from kostyl.ml.configs import ScheduledParamConfig
from kostyl.utils import setup_logger

from .schedulers import SCHEDULER_MAPPING
from .schedulers import CosineScheduler
from .schedulers import LinearScheduler
from .schedulers import PlateauWithAnnealingScheduler


logger = setup_logger(fmt="only_message")


class OVERRIDABLE_CONFIG_KWARGS(TypedDict, total=False):  # noqa: D101, N801
    scheduler_type: SCHEDULER
    freeze_ratio: float | None
    warmup_ratio: float | None
    warmup_value: float | None
    base_value: float
    final_value: float | None
    plateau_ratio: float | None


def create_scheduler(
    config: ScheduledParamConfig,
    param_group_field: str,
    num_iters: int,
    optim: Optimizer,
    multiplier_field: str | None = None,
    skip_if_zero: bool = False,
    apply_if_field: str | None = None,
    ignore_if_field: str | None = None,
    **kwargs: Unpack[OVERRIDABLE_CONFIG_KWARGS],
) -> LinearScheduler | CosineScheduler | PlateauWithAnnealingScheduler:
    """
    Converts a ScheduledParamConfig to a scheduler instance.

    Args:
        config: Configuration object for the scheduler.
        param_group_field: The field name in the optimizer's param groups to schedule.
        num_iters: Total number of iterations.
        optim: The optimizer instance.
        multiplier_field: Optional per-group field name that contains a multiplier applied to the scheduled value. If None, no multiplier is applied.
        skip_if_zero: Leave groups untouched when their target field equals zero.
            Default is False.
        apply_if_field: Require this key to be present in a param group before updating.
        ignore_if_field: Skip groups that declare this key in their dictionaries.
        **kwargs: Optional overrides for scheduler configuration parameters (e.g., base_value,
            final_value, warmup_ratio, etc.). These overrides take precedence over the values
            provided in the `config` object.

    Returns:
        A scheduler instance based on the configuration.

    """
    scheduler_type = kwargs.get("scheduler_type", config.scheduler_type)
    base_value = kwargs.get("base_value", config.base_value)
    final_value = kwargs.get("final_value", config.final_value)
    warmup_ratio = kwargs.get("warmup_ratio", config.warmup_ratio)
    warmup_value = kwargs.get("warmup_value", config.warmup_value)
    freeze_ratio = kwargs.get("freeze_ratio", config.freeze_ratio)
    plateau_ratio = kwargs.get("plateau_ratio", config.plateau_ratio)

    if scheduler_type is None:
        raise ValueError("scheduler_type must be specified in the config.")

    if "plateau" in scheduler_type:
        lookup_scheduler_type = "plateau"
    else:
        lookup_scheduler_type = scheduler_type

    scheduler_cls = SCHEDULER_MAPPING[lookup_scheduler_type]  # type: ignore

    if issubclass(scheduler_cls, PlateauWithAnnealingScheduler):
        if "cosine" in scheduler_type:
            annealing_type = "cosine"
        elif "linear" in scheduler_type:
            annealing_type = "linear"
        else:
            raise ValueError(f"Unknown annealing_type: {scheduler_type}")
        scheduler = scheduler_cls(
            optimizer=optim,
            param_group_field=param_group_field,
            num_iters=num_iters,
            plateau_value=base_value,
            final_value=final_value,  # type: ignore
            warmup_ratio=warmup_ratio,
            warmup_value=warmup_value,
            freeze_ratio=freeze_ratio,
            plateau_ratio=plateau_ratio,  # type: ignore
            annealing_type=annealing_type,
            multiplier_field=multiplier_field,
            skip_if_zero=skip_if_zero,
            apply_if_field=apply_if_field,
            ignore_if_field=ignore_if_field,
        )
    elif issubclass(scheduler_cls, LinearScheduler):
        scheduler = scheduler_cls(
            optimizer=optim,
            param_group_field=param_group_field,
            num_iters=num_iters,
            initial_value=base_value,
            final_value=final_value,  # type: ignore
            multiplier_field=multiplier_field,
            skip_if_zero=skip_if_zero,
            apply_if_field=apply_if_field,
            ignore_if_field=ignore_if_field,
        )
    elif issubclass(scheduler_cls, CosineScheduler):
        scheduler = scheduler_cls(
            optimizer=optim,
            param_group_field=param_group_field,
            num_iters=num_iters,
            base_value=base_value,
            final_value=final_value,  # type: ignore
            warmup_ratio=warmup_ratio,
            warmup_value=warmup_value,
            freeze_ratio=freeze_ratio,
            multiplier_field=multiplier_field,
            skip_if_zero=skip_if_zero,
            apply_if_field=apply_if_field,
            ignore_if_field=ignore_if_field,
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    return scheduler


def create_optimizer(  # noqa: C901
    parameters_groups: ParamsT,
    optimizer_config: OPTIMIZER_CONFIG,
    lr: float,
    weight_decay: float,
) -> Optimizer:
    """
    Creates an optimizer based on the configuration.

    Args:
        parameters_groups: Parameter groups for the optimizer.
        optimizer_config: Configuration for the optimizer.
        lr: Learning rate.
        weight_decay: Weight decay.

    Returns:
        An instantiated optimizer.

    """
    if isinstance(optimizer_config, AdamConfig):
        match optimizer_config.type:
            case "Adam":
                from torch.optim import Adam

                optimizer = Adam(
                    params=parameters_groups,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=optimizer_config.betas,
                )

            case "AdamW":
                from torch.optim import AdamW

                optimizer = AdamW(
                    params=parameters_groups,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=optimizer_config.betas,
                )
                return optimizer
            case _:
                raise ValueError(f"Unsupported optimizer type: {optimizer_config.type}")
    elif isinstance(optimizer_config, MuonConfig):
        from torch.optim import Muon

        optimizer = Muon(
            params=parameters_groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=optimizer_config.momentum,
            nesterov=optimizer_config.nesterov,
            ns_coefficients=optimizer_config.ns_coefficients,
            ns_steps=optimizer_config.ns_steps,
        )
    elif isinstance(optimizer_config, AdamWithPrecisionConfig):
        try:
            import torchao  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "torchao is required for low-precision Adam optimizers. "
                "Please install it via 'pip install torchao'."
            ) from e
        match optimizer_config.type:
            case "Adam8bit":
                from torchao.optim import Adam8bit

                logger.warning(
                    "Ignoring weight_decay for Adam8bit optimizer as it is not supported."
                )

                optimizer = Adam8bit(
                    params=parameters_groups,
                    lr=lr,
                    betas=optimizer_config.betas,
                    block_size=optimizer_config.block_size,
                    bf16_stochastic_round=optimizer_config.bf16_stochastic_round,
                )
            case "Adam4bit":
                from torchao.optim import Adam4bit

                logger.warning(
                    "Ignoring weight_decay for Adam4bit optimizer as it is not supported."
                )

                optimizer = Adam4bit(
                    params=parameters_groups,
                    lr=lr,
                    betas=optimizer_config.betas,
                    block_size=optimizer_config.block_size,
                    bf16_stochastic_round=optimizer_config.bf16_stochastic_round,
                )
            case "AdamFp8":
                from torchao.optim import AdamFp8

                logger.warning(
                    "Ignoring weight_decay for AdamFp8 optimizer as it is not supported."
                )

                optimizer = AdamFp8(
                    params=parameters_groups,
                    lr=lr,
                    betas=optimizer_config.betas,
                    block_size=optimizer_config.block_size,
                    bf16_stochastic_round=optimizer_config.bf16_stochastic_round,
                )
            case "AdamW8bit":
                from torchao.optim import AdamW8bit

                optimizer = AdamW8bit(
                    params=parameters_groups,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=optimizer_config.betas,
                    block_size=optimizer_config.block_size,
                    bf16_stochastic_round=optimizer_config.bf16_stochastic_round,
                )
            case "AdamW4bit":
                from torchao.optim import AdamW4bit

                optimizer = AdamW4bit(
                    params=parameters_groups,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=optimizer_config.betas,
                    block_size=optimizer_config.block_size,
                    bf16_stochastic_round=optimizer_config.bf16_stochastic_round,
                )
            case "AdamWFp8":
                from torchao.optim import AdamWFp8

                optimizer = AdamWFp8(
                    params=parameters_groups,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=optimizer_config.betas,
                    block_size=optimizer_config.block_size,
                    bf16_stochastic_round=optimizer_config.bf16_stochastic_round,
                )
            case _:
                raise ValueError(f"Unsupported optimizer type: {optimizer_config.type}")
    else:
        raise ValueError("Unsupported optimizer configuration type.")
    return optimizer
