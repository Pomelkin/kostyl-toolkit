import torch
from torch import nn

from kostyl.ml.param_groups import create_param_groups


LR = 1e-3
WEIGHT_DECAY = 0.01


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(4, 8)
        self.linear = nn.Linear(8, 8)
        self.ln = nn.LayerNorm(8)


def collect_params(groups: list[dict]) -> list[torch.nn.Parameter]:
    return [p for g in groups for p in g["params"]]


class TestCreateParamGroups:
    def test_no_decay_defaults(self) -> None:
        model = TinyModel()
        groups = create_param_groups(model, weight_decay=WEIGHT_DECAY, lr=LR)

        decay_groups = [g for g in groups if g["weight_decay"] == WEIGHT_DECAY]
        no_decay_groups = [g for g in groups if g["weight_decay"] == 0.0]

        # Only linear.weight should keep weight decay; emb, ln, and biases should not
        assert len(collect_params(decay_groups)) == 1
        assert collect_params(decay_groups)[0] is model.linear.weight
        expected_no_decay = len(list(model.parameters())) - 1
        assert len(collect_params(no_decay_groups)) == expected_no_decay

    def test_all_groups_receive_lr(self) -> None:
        model = TinyModel()
        groups = create_param_groups(model, weight_decay=WEIGHT_DECAY, lr=LR)
        assert all(g["lr"] == LR for g in groups)

    def test_no_lr_keywords_zero_out_lr(self) -> None:
        model = TinyModel()
        groups = create_param_groups(
            model,
            weight_decay=WEIGHT_DECAY,
            lr=LR,
            no_lr_keywords={"emb"},
        )
        frozen_lr_groups = [g for g in groups if g["lr"] == 0.0]
        assert collect_params(frozen_lr_groups) == [model.emb.weight]

    def test_extra_no_decay_keywords_extend_defaults(self) -> None:
        model = TinyModel()
        groups = create_param_groups(
            model,
            weight_decay=WEIGHT_DECAY,
            lr=LR,
            no_decay_keywords={"linear"},
        )
        # The default keywords still apply, so now every parameter is no-decay
        assert all(g["weight_decay"] == 0.0 for g in groups)

    def test_frozen_params_are_excluded(self) -> None:
        model = TinyModel()
        model.emb.weight.requires_grad = False
        groups = create_param_groups(model, weight_decay=WEIGHT_DECAY, lr=LR)
        assert all(p is not model.emb.weight for p in collect_params(groups))

    def test_groups_are_fused_by_settings(self) -> None:
        model = TinyModel()
        groups = create_param_groups(model, weight_decay=WEIGHT_DECAY, lr=LR)

        # Groups are fused: one per unique (lr, weight_decay) combination
        settings = [(g["lr"], g["weight_decay"]) for g in groups]
        assert len(settings) == len(set(settings))

        # No parameters are lost during fusing
        assert len(collect_params(groups)) == len(list(model.parameters()))

    def test_groups_are_valid_for_torch_optimizer(self) -> None:
        model = TinyModel()
        groups = create_param_groups(model, weight_decay=WEIGHT_DECAY, lr=LR)
        optimizer = torch.optim.AdamW(groups)
        assert len(optimizer.param_groups) == len(groups)
