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
        initial_value: float,
        final_value: float,
    ) -> None:
        self.param_name = param_name
        self.num_iters = num_iters
        self.initial_value = initial_value
        self.final_value = final_value

        self.scheduled_values: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.current_value_ = self.initial_value
        return

    def _create_scheduler(self) -> None:
        self.scheduled_values = np.linspace(
            self.initial_value, self.final_value, num=self.num_iters, dtype=np.float64
        )
        self._verify()
        return

    @override
    def _verify(self) -> None:
        if len(self.scheduled_values) != self.num_iters:
            raise ValueError(
                f"Scheduler length ({len(self.scheduled_values)}) does not match total_iters ({self.num_iters})."
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


class LinearScheduler(_LinearScheduleBase):
    """Implements a linear scheduler for adjusting parameter values in torch.optim.Optimizer."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        param_group_field: str,
        num_iters: int,
        initial_value: float,
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
            initial_value: Value used on the first iteration.
            final_value: Value used once ``num_iters`` iterations are consumed.
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
            initial_value=initial_value,
            final_value=final_value,
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

            if self.multiplier_field is not None:
                multiplier = pg.get(self.multiplier_field, 1.0)
                value = value * multiplier

            if isinstance(current_param_val, torch.Tensor):
                current_param_val.fill_(value)
            else:
                pg[self.param_group_field] = value
        return


class LinearParamScheduler(_LinearScheduleBase):
    """LinearParamScheduler adjusts a parameter value using a linear scheduler."""

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
        self.scheduled_values = np.array([], dtype=np.float64)
        return

    @override
    def state_dict(self) -> dict[str, Any]:
        state = {k: v for k, v in self.__dict__.items() if k != "scheduled_values"}
        return state

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
