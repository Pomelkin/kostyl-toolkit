from typing import Any
from typing import Literal
from typing import override

import numpy as np
import torch

from .base import BaseScheduler


class _PlateauWithAnnealingCore(BaseScheduler):
    """Core annealing with plateau scheduler logic."""

    def __init__(
        self,
        param_name: str,
        num_iters: int,
        plateau_value: float,
        final_value: float,
        plateau_ratio: float,
        warmup_value: float | None = None,
        warmup_ratio: float | None = None,
        freeze_ratio: float | None = None,
        annealing_type: Literal["cosine", "linear"] = "cosine",
    ) -> None:
        if annealing_type not in ("cosine", "linear"):
            raise ValueError(
                f"Annealing type must be 'cosine' or 'linear', got {annealing_type}."
            )
        if warmup_ratio is not None and not (0 < warmup_ratio < 1):
            raise ValueError(f"Warmup ratio must be in (0, 1), got {warmup_ratio}.")
        if (warmup_value is None) != (warmup_ratio is None):
            raise ValueError(
                "Both warmup_ratio and warmup_value must be provided or neither."
            )
        if freeze_ratio is not None and not (0 < freeze_ratio < 1):
            raise ValueError(f"Freeze ratio must be in (0, 1), got {freeze_ratio}.")
        if not (0 < plateau_ratio < 1):
            raise ValueError(f"Plateau ratio must be in (0, 1), got {plateau_ratio}.")

        pre_annealing_ratio = plateau_ratio + (warmup_ratio or 0) + (freeze_ratio or 0)
        if pre_annealing_ratio > 1:
            raise ValueError(
                "The sum of plateau_ratio, warmup_ratio, and freeze_ratio must <= 1, got "
                f"{pre_annealing_ratio}."
            )

        self.param_name = param_name
        self.num_iters = num_iters
        self.plateau_value = plateau_value
        self.final_value = final_value
        self.warmup_ratio = warmup_ratio
        self.warmup_value = warmup_value
        self.plateau_ratio = plateau_ratio
        self.freeze_ratio = freeze_ratio
        self.annealing_type = annealing_type

        self.scheduled_values: np.ndarray = np.array([], dtype=np.float64)
        self.current_value_ = self.plateau_value
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
                self.warmup_value, self.plateau_value, warmup_iters, dtype=np.float64
            )
        else:
            warmup_iters = 0
            warmup_schedule = np.array([], dtype=np.float64)

        # Create plateau schedule
        plateau_iters = int(self.num_iters * self.plateau_ratio)
        plateau_schedule = np.full(plateau_iters, self.plateau_value, dtype=np.float64)

        # Create annealing schedule
        annealing_iters = self.num_iters - freeze_iters - warmup_iters - plateau_iters
        if annealing_iters > 0:
            match self.annealing_type:
                case "cosine":
                    iters = np.arange(annealing_iters, dtype=np.float64)
                    annealing_schedule = self.final_value + 0.5 * (
                        self.plateau_value - self.final_value
                    ) * (1 + np.cos(np.pi * iters / len(iters)))
                case "linear":
                    annealing_schedule = np.linspace(
                        self.plateau_value,
                        self.final_value,
                        annealing_iters,
                        dtype=np.float64,
                    )
                case _:
                    raise ValueError(
                        f"Unsupported annealing type: {self.annealing_type}"
                    )
        else:
            annealing_schedule = np.array([], dtype=np.float64)

        # Concatenate all parts of the schedule
        self.scheduled_values = np.concatenate(
            (
                freeze_schedule,
                warmup_schedule,
                plateau_schedule,
                annealing_schedule,
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
    def step(self, it: int) -> float | None:
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


class PlateauWithAnnealingScheduler(_PlateauWithAnnealingCore):
    """
    Applies an annealing schedule with plateau to an optimizer param-group field.

    Schedule phases: freeze (0) → warmup → plateau (plateau_value) → annealing (cosine/linear) to final_value.
    The plateau phase maintains the plateau_value before annealing begins.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        param_group_field: str,
        num_iters: int,
        plateau_value: float,
        final_value: float,
        plateau_ratio: float,
        warmup_value: float | None = None,
        warmup_ratio: float | None = None,
        freeze_ratio: float | None = None,
        annealing_type: Literal["cosine", "linear"] = "cosine",
        multiplier_field: str | None = None,
        skip_if_zero: bool = False,
        apply_if_field: str | None = None,
        ignore_if_field: str | None = None,
    ) -> None:
        """
        Configure annealing scheduling for matching optimizer groups.

        Args:
            optimizer: Optimizer whose param groups are updated in-place.
            param_group_field: Name of the field that receives the scheduled value.
            num_iters: Number of scheduler iterations before clamping at ``final_value``.
            plateau_value: Value maintained during plateau phase and used as annealing start.
            final_value: Value approached as iterations progress during annealing.
            plateau_ratio: Fraction of iterations to maintain ``plateau_value`` before annealing.
            warmup_ratio: Optional fraction of iterations to linearly ramp from ``warmup_value`` to ``plateau_value``.
            warmup_value: Starting value for the warmup ramp.
            freeze_ratio: Optional fraction of iterations to keep the value frozen at zero at the beginning.
            annealing_type: Type of annealing from plateau to final value ("cosine" or "linear").
            multiplier_field: Optional per-group multiplier applied to the scheduled value.
                if specified, but not found in a group, multiplier of 1.0 is assumed for that group.
            skip_if_zero: Leave groups untouched when their target field equals zero.
            apply_if_field: Require this key to be present in a param group before updating.
            ignore_if_field: Skip groups that declare this key in their dictionaries.

        """
        self.apply_if_field = apply_if_field
        self.ignore_if_field = ignore_if_field
        self.optimizer = optimizer
        self.multiplier_field = multiplier_field
        self.skip_if_zero = skip_if_zero
        super().__init__(
            param_name=param_group_field,
            num_iters=num_iters,
            plateau_value=plateau_value,
            final_value=final_value,
            plateau_ratio=plateau_ratio,
            warmup_ratio=warmup_ratio,
            warmup_value=warmup_value,
            freeze_ratio=freeze_ratio,
            annealing_type=annealing_type,
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
    def step(self, it: int) -> None:  # noqa: C901
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

            current_param_val = pg[self.param_group_field]

            if self.skip_if_zero:
                if isinstance(current_param_val, torch.Tensor):
                    if current_param_val.item() == 0:
                        continue
                elif current_param_val == 0:
                    continue

            pg_value = value
            if self.multiplier_field is not None:
                multiplier = pg.get(self.multiplier_field, 1.0)
                pg_value = pg_value * multiplier

            if isinstance(current_param_val, torch.Tensor):
                current_param_val.fill_(pg_value)
            else:
                pg[self.param_group_field] = pg_value
        return


class PlateauWithAnnealingParamScheduler(_PlateauWithAnnealingCore):
    """
    Standalone annealing scheduler with plateau for non-optimizer parameters.

    Schedule phases: freeze (0) → warmup → plateau (plateau_value) → annealing (cosine/linear) to final_value.
    The plateau phase maintains the plateau_value before annealing begins.
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
