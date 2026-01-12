from typing import Any
from typing import override

import numpy as np
import torch

from .base import BaseScheduler


class _CosineWithPlateauSchedulerCore(BaseScheduler):
    """Core cosine with plateau scheduler logic."""

    def __init__(
        self,
        param_name: str,
        num_iters: int,
        base_value: float,
        final_value: float,
        plateau_ratio: float,
        warmup_value: float | None = None,
        warmup_ratio: float | None = None,
        freeze_ratio: float | None = None,
    ) -> None:
        if warmup_ratio is not None:
            if not (0 < warmup_ratio < 1):
                raise ValueError(f"Warmup ratio must be in (0, 1), got {warmup_ratio}.")
        if (warmup_value is None) != (warmup_ratio is None):
            raise ValueError(
                "Both warmup_ratio and warmup_value must be provided or neither."
            )
        if freeze_ratio is not None:
            if not (0 < freeze_ratio < 1):
                raise ValueError(f"Freeze ratio must be in (0, 1), got {freeze_ratio}.")
        if not (0 < plateau_ratio < 1):
            raise ValueError(f"Plateau ratio must be in (0, 1), got {plateau_ratio}.")

        pre_annealing_ratio = (
            plateau_ratio
            + (warmup_ratio if warmup_ratio is not None else 0)
            + (freeze_ratio if freeze_ratio is not None else 0)
        )
        if pre_annealing_ratio > 1:
            raise ValueError(
                "The sum of plateau_ratio, warmup_ratio, and freeze_ratio must <= 1, got "
                f"{pre_annealing_ratio}."
            )

        self.param_name = param_name
        self.num_iters = num_iters
        self.base_value = base_value
        self.final_value = final_value
        self.cosine_annealing_ratio = 1 - pre_annealing_ratio
        self.plateau_ratio = plateau_ratio
        self.warmup_ratio = warmup_ratio
        self.warmup_value = warmup_value
        self.freeze_ratio = freeze_ratio

        self.scheduled_values: np.ndarray = np.array([], dtype=np.float64)
        self.current_value_ = self.base_value
        return

    def _create_scheduler(self) -> None:
        # Create freeze schedule
        if self.freeze_ratio is not None:
            freeze_iters = int(self.num_iters * self.freeze_ratio)
            freeze_schedule = np.zeros(freeze_iters, dtype=np.float64)
        else:
            freeze_iters = 0
            freeze_schedule = np.array([], dtype=np.float64)

        # Create linear warmup schedule
        if self.warmup_ratio is not None and self.warmup_value is not None:
            warmup_iters = int(self.num_iters * self.warmup_ratio)
            warmup_schedule = np.linspace(
                self.warmup_value, self.base_value, warmup_iters, dtype=np.float64
            )
        else:
            warmup_iters = 0
            warmup_schedule = np.array([], dtype=np.float64)

        # Create cosine annealing schedule
        if self.cosine_annealing_ratio > 0:
            cosine_annealing_iters = int(self.num_iters * self.cosine_annealing_ratio)
            iters = np.arange(cosine_annealing_iters)
            cosine_annealing_schedule = self.final_value + 0.5 * (
                self.base_value - self.final_value
            ) * (1 + np.cos(np.pi * iters / len(iters)))
        else:
            cosine_annealing_iters = 0
            cosine_annealing_schedule = np.array([], dtype=np.float64)

        plateau_iters = (
            self.num_iters - warmup_iters - freeze_iters - cosine_annealing_iters
        )
        if plateau_iters > 0:
            plateau_schedule = np.full(plateau_iters, self.base_value, dtype=np.float64)
        else:
            plateau_schedule = np.array([], dtype=np.float64)

        # Concatenate all parts of the schedule
        self.scheduled_values = np.concatenate(
            (
                freeze_schedule,
                warmup_schedule,
                plateau_schedule,
                cosine_annealing_schedule,
            )
        )
        self._verify()
        return

    @override
    def _verify(self) -> None:
        if len(self.scheduled_values) != self.num_iters:
            raise ValueError(
                f"Scheduler length ({len(self.scheduled_values)}) does not match num_iters ({self.num_iters})."
            )
        return

    @override
    def step(self, it: int) -> None | float:
        raise NotImplementedError

    def _get_value(self, it: int) -> float:
        if len(self.scheduled_values) == 0:
            self._create_scheduler()

        if it >= self.num_iters:
            value: float = self.final_value
        else:
            value: float = self.scheduled_values[it]
        self.current_value_ = value
        return value

    @override
    def current_value(self) -> dict[str, float]:
        return {self.param_name: self.current_value_}


class CosineWithPlateuScheduler(_CosineWithPlateauSchedulerCore):
    """
    Applies a cosine schedule with plateau to an optimizer param-group field.

    Schedule phases: freeze (0) → warmup → plateau (base_value) → cosine annealing to final_value.
    The plateau phase maintains the base_value before cosine annealing begins.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        param_group_field: str,
        num_iters: int,
        base_value: float,
        final_value: float,
        plateau_ratio: float,
        warmup_value: float | None = None,
        warmup_ratio: float | None = None,
        freeze_ratio: float | None = None,
        multiplier_field: str | None = None,
        skip_if_zero: bool = False,
        apply_if_field: str | None = None,
        ignore_if_field: str | None = None,
    ) -> None:
        """
        Configure cosine scheduling for matching optimizer groups.

        Args:
            optimizer: Optimizer whose param groups are updated in-place.
            param_group_field: Name of the field that receives the scheduled value.
            num_iters: Number of scheduler iterations before clamping at ``final_value``.
            base_value: Value maintained during plateau phase and used as cosine start.
            final_value: Value approached as iterations progress during cosine annealing.
            plateau_ratio: Fraction of iterations to maintain ``base_value`` before cosine annealing.
            warmup_ratio: Optional fraction of iterations to linearly ramp from ``warmup_value`` to ``base_value``.
            warmup_value: Starting value for the warmup ramp.
            freeze_ratio: Optional fraction of iterations to keep the value frozen at zero at the beginning.
            multiplier_field: Optional per-group multiplier applied to the scheduled value.
            skip_if_zero: Leave groups untouched when their target field equals zero.
            apply_if_field: Require this flag to be present in a param group before updating.
            ignore_if_field: Skip groups that declare this flag.

        """
        self.apply_if_field = apply_if_field
        self.ignore_if_field = ignore_if_field
        self.optimizer = optimizer
        self.multiplier_field = multiplier_field
        self.skip_if_zero = skip_if_zero
        super().__init__(
            param_name=param_group_field,
            num_iters=num_iters,
            base_value=base_value,
            final_value=final_value,
            plateau_ratio=plateau_ratio,
            warmup_ratio=warmup_ratio,
            warmup_value=warmup_value,
            freeze_ratio=freeze_ratio,
        )
        self.param_group_field = param_group_field
        return

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
        self.scheduled_values = np.array([], dtype=np.float64)
        return

    @override
    def state_dict(self) -> dict[str, Any]:
        state = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["scheduled_values", "optimizer"]
        }
        return state

    @override
    def step(self, it: int) -> None:
        value = self._get_value(it)
        for pg in self.optimizer.param_groups:
            if self.param_group_field not in pg:
                raise ValueError(
                    f"Parameter group field '{self.param_group_field}' not found in optimizer parameter groups."
                )

            if (self.apply_if_field is not None) and (self.apply_if_field not in pg):
                continue

            if (self.ignore_if_field is not None) and (self.ignore_if_field in pg):
                continue

            if self.skip_if_zero and pg[self.param_group_field] == 0:
                continue

            if self.multiplier_field is not None:
                if self.multiplier_field not in pg:
                    multiplier = 1.0
                else:
                    multiplier = pg[self.multiplier_field]
                pg[self.param_group_field] = value * multiplier
            else:
                pg[self.param_group_field] = value
        return


class CosineWithPlateauParamScheduler(_CosineWithPlateauSchedulerCore):
    """
    Standalone cosine scheduler with plateau for non-optimizer parameters.

    Schedule phases: freeze (0) → warmup → plateau (base_value) → cosine annealing to final_value.
    The plateau phase maintains the base_value before cosine annealing begins.
    """

    @override
    def step(self, it: int) -> float:
        """
        Computes the value corresponding to the given iteration step.

        Args:
            it: The current iteration index used for value computation.

        Returns:
            The computed value for the provided iteration step as a float.

        """
        value = self._get_value(it)
        return value

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
        self.scheduled_values = np.array([], dtype=np.float64)
        return

    @override
    def state_dict(self) -> dict[str, Any]:
        state = {k: v for k, v in self.__dict__.items() if k != "scheduled_values"}
        return state
