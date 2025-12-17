from typing import Any
from typing import override

import numpy as np
import numpy.typing as npt
import torch

from .base import BaseScheduler


class _LinearScheduleBase(BaseScheduler):
    def __init__(
        self,
        param_name: str,
        num_iters: int,
        start_value: float,
        final_value: float,
    ) -> None:
        self.param_name = param_name
        self.num_iters = num_iters
        self.start_value = start_value
        self.final_value = final_value

        self.scheduler_values: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.current_value_ = self.start_value
        return

    def _create_scheduler(self) -> None:
        self.scheduler_values = np.linspace(
            self.start_value, self.final_value, num=self.num_iters, dtype=np.float64
        )
        if len(self.scheduler_values) != self.num_iters:
            raise ValueError(
                f"Scheduler length ({len(self.scheduler_values)}) does not match total_iters ({self.num_iters})."
            )
        return

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.scheduler_values = np.array([], dtype=np.float64)
        return

    @override
    def step(self, it: int) -> None | float:
        raise NotImplementedError

    def _get_value(self, it: int) -> float:
        if len(self.scheduler_values) == 0:
            self._create_scheduler()

        if it >= self.num_iters:
            value: float = self.final_value
        else:
            value: float = self.scheduler_values[it]
        self.current_value_ = value
        return value

    @override
    def current_value(self) -> dict[str, float]:
        return {self.param_name: self.current_value_}


class LinearScheduler(_LinearScheduleBase):
    """Implements a linear scheduler for adjusting parameter values in torch.optim.Optimizer."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        param_group_field: str,
        num_iters: int,
        start_value: float,
        final_value: float,
        multiplier_field: str | None = None,
        skip_if_zero: bool = False,
        apply_if_field: str | None = None,
        ignore_if_field: str | None = None,
    ) -> None:
        """
        Configure which optimizer groups get a linear value schedule.

        Args:
            optimizer: Optimizer whose param groups are updated in-place.
            param_group_field: Name of the field that receives the scheduled value.
            num_iters: Number of scheduler iterations before clamping at ``final_value``.
            start_value: Value used on the first iteration.
            final_value: Value used once ``num_iters`` iterations are consumed.
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
            start_value=start_value,
            final_value=final_value,
        )
        self.param_group_field = param_group_field
        return

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


class LinearParamScheduler(_LinearScheduleBase):
    """LinearParamScheduler adjusts a parameter value using a linear scheduler."""

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
