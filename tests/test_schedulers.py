from typing import Any
from itertools import pairwise

import numpy as np
import pytest
import torch
from torch.optim import SGD
from torch.optim import Optimizer

from kostyl.ml.configs import Lr
from kostyl.ml.optim import create_scheduler
from kostyl.ml.optim.schedulers import CosineScheduler
from kostyl.ml.optim.schedulers import LinearParamScheduler
from kostyl.ml.optim.schedulers import LinearScheduler
from kostyl.ml.optim.schedulers import PlateauWithAnnealingScheduler


NUM_ITERS = 10


def make_optimizer(groups: list[dict[str, Any]] | None = None) -> Optimizer:
    if groups is None:
        groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1.0}]
    return SGD(groups)


def get_lrs(optim: Optimizer) -> list[float]:
    return [pg["lr"] for pg in optim.param_groups]


class TestLinearScheduler:
    def test_annealing_endpoints(self) -> None:
        optim = make_optimizer()
        scheduler = LinearScheduler(
            optimizer=optim,
            param_group_field="lr",
            num_iters=NUM_ITERS,
            base_value=1.0,
            final_value=0.0,
        )

        scheduler.step(0)
        assert get_lrs(optim) == [1.0]

        scheduler.step(NUM_ITERS - 1)
        assert get_lrs(optim) == [0.0]

    def test_clamps_to_final_value_after_num_iters(self) -> None:
        optim = make_optimizer()
        scheduler = LinearScheduler(
            optimizer=optim,
            param_group_field="lr",
            num_iters=NUM_ITERS,
            base_value=1.0,
            final_value=0.5,
        )
        scheduler.step(NUM_ITERS * 100)
        assert get_lrs(optim) == [0.5]

    def test_warmup_starts_from_warmup_value(self) -> None:
        optim = make_optimizer()
        scheduler = LinearScheduler(
            optimizer=optim,
            param_group_field="lr",
            num_iters=NUM_ITERS,
            base_value=1.0,
            final_value=0.0,
            warmup_ratio=0.2,
            warmup_value=0.1,
        )

        scheduler.step(0)
        assert get_lrs(optim) == [0.1]

        scheduler.step(1)
        assert get_lrs(optim) == [1.0]

    def test_freeze_keeps_value_at_zero(self) -> None:
        optim = make_optimizer()
        scheduler = LinearScheduler(
            optimizer=optim,
            param_group_field="lr",
            num_iters=NUM_ITERS,
            base_value=1.0,
            final_value=0.0,
            freeze_ratio=0.2,
        )
        scheduler.step(0)
        assert get_lrs(optim) == [0.0]

    def test_warmup_requires_both_parameters(self) -> None:
        with pytest.raises(ValueError, match="warmup_ratio and warmup_value"):
            LinearScheduler(
                optimizer=make_optimizer(),
                param_group_field="lr",
                num_iters=NUM_ITERS,
                base_value=1.0,
                final_value=0.0,
                warmup_ratio=0.2,
            )

    def test_multiplier_field(self) -> None:
        groups = [
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1.0},
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1.0, "mult": 0.5},
        ]
        optim = make_optimizer(groups)
        scheduler = LinearScheduler(
            optimizer=optim,
            param_group_field="lr",
            num_iters=NUM_ITERS,
            base_value=1.0,
            final_value=0.0,
            multiplier_field="mult",
        )
        scheduler.step(0)
        assert get_lrs(optim) == [1.0, 0.5]

    def test_skip_if_zero(self) -> None:
        groups = [
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1.0},
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 0.0},
        ]
        optim = make_optimizer(groups)
        scheduler = LinearScheduler(
            optimizer=optim,
            param_group_field="lr",
            num_iters=NUM_ITERS,
            base_value=0.5,
            final_value=0.0,
            skip_if_zero=True,
        )
        scheduler.step(0)
        assert get_lrs(optim) == [0.5, 0.0]

    def test_apply_and_ignore_fields(self) -> None:
        groups = [
            {
                "params": [torch.nn.Parameter(torch.zeros(1))],
                "lr": 1.0,
                "use_sched": True,
            },
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1.0},
        ]
        optim = make_optimizer(groups)
        scheduler = LinearScheduler(
            optimizer=optim,
            param_group_field="lr",
            num_iters=NUM_ITERS,
            base_value=0.5,
            final_value=0.0,
            apply_if_field="use_sched",
        )
        scheduler.step(0)
        assert get_lrs(optim) == [0.5, 1.0]

    def test_state_dict_roundtrip(self) -> None:
        optim = make_optimizer()
        scheduler = LinearScheduler(
            optimizer=optim,
            param_group_field="lr",
            num_iters=NUM_ITERS,
            base_value=1.0,
            final_value=0.0,
        )
        scheduler.step(3)
        state = scheduler.state_dict()

        assert "optimizer" not in state
        assert "scheduled_values" not in state

        restored = LinearScheduler(
            optimizer=optim,
            param_group_field="lr",
            num_iters=1,
            base_value=0.0,
            final_value=0.0,
        )
        restored.load_state_dict(state)

        assert restored.num_iters == NUM_ITERS
        assert restored.current_value() == scheduler.current_value()

        restored.step(3)
        assert get_lrs(optim) == pytest.approx([scheduler.current_value()["lr"]])


class TestCosineScheduler:
    def test_schedule_shape(self) -> None:
        optim = make_optimizer()
        scheduler = CosineScheduler(
            optimizer=optim,
            param_group_field="lr",
            num_iters=NUM_ITERS,
            base_value=1.0,
            final_value=0.1,
        )
        values = []
        for it in range(NUM_ITERS):
            scheduler.step(it)
            values.append(get_lrs(optim)[0])

        assert values[0] == pytest.approx(1.0)
        assert values[-1] == pytest.approx(0.1, abs=0.05)
        assert all(a >= b for a, b in pairwise(values))
        assert np.all(np.asarray(values) >= 0.1 - 1e-9)


class TestLinearParamScheduler:
    def test_step_returns_value_without_optimizer(self) -> None:
        scheduler = LinearParamScheduler(
            param_name="weight_decay",
            num_iters=NUM_ITERS,
            base_value=0.1,
            final_value=0.0,
        )
        assert scheduler.step(0) == pytest.approx(0.1)
        assert scheduler.step(NUM_ITERS - 1) == pytest.approx(0.0)
        assert scheduler.current_value() == {"weight_decay": 0.0}


class TestCreateSchedulerFactory:
    def test_creates_cosine_scheduler_from_config(self) -> None:
        config = Lr(
            scheduler_type="cosine",
            base_value=3e-4,
            final_value=3e-5,
            warmup_ratio=0.05,
            warmup_value=1e-6,
        )
        scheduler = create_scheduler(
            config=config,
            param_group_field="lr",
            num_iters=NUM_ITERS,
            optim=make_optimizer(),
        )
        assert isinstance(scheduler, CosineScheduler)
        assert scheduler.base_value == pytest.approx(3e-4)

    def test_creates_plateau_scheduler_from_config(self) -> None:
        config = Lr(
            scheduler_type="plateau-with-cosine-annealing",
            base_value=1e-3,
            final_value=1e-5,
            plateau_ratio=0.5,
        )
        scheduler = create_scheduler(
            config=config,
            param_group_field="lr",
            num_iters=NUM_ITERS,
            optim=make_optimizer(),
        )
        assert isinstance(scheduler, PlateauWithAnnealingScheduler)

    def test_kwargs_override_config(self) -> None:
        config = Lr(scheduler_type="cosine", base_value=1e-3, final_value=1e-5)
        scheduler = create_scheduler(
            config=config,
            param_group_field="lr",
            num_iters=NUM_ITERS,
            optim=make_optimizer(),
            base_value=5e-4,
        )
        assert isinstance(scheduler, CosineScheduler)
        assert scheduler.base_value == pytest.approx(5e-4)

    def test_missing_scheduler_type_raises(self) -> None:
        config = Lr(base_value=1e-3)
        with pytest.raises(ValueError, match="scheduler_type"):
            create_scheduler(
                config=config,
                param_group_field="lr",
                num_iters=NUM_ITERS,
                optim=make_optimizer(),
            )
