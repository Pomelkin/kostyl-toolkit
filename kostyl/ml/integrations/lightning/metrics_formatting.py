from collections.abc import Mapping
from typing import TypeVar

import torch.distributed as dist
from torch import Tensor
from torchmetrics import Metric


MetricTypeT = TypeVar("MetricTypeT", bound=Metric | Tensor | int | float)


def format_metric_names(
    metrics: Mapping[str, MetricTypeT],
    prefix: str | None = None,
    suffix: str | None = None,
    add_dist_rank: bool = False,
) -> Mapping[str, MetricTypeT]:
    """Add prefix, suffix, and optionally distributed rank to metric names."""
    if prefix is None and suffix is None:
        return metrics

    if prefix is None:
        prefix = ""
    elif prefix.endswith("/"):
        prefix = prefix[:-1]

    if suffix is None:
        suffix = ""
    elif suffix.startswith("/"):
        suffix = suffix[1:]

    new_metrics_dict: dict[str, MetricTypeT] = {}
    for key, value in metrics.items():
        new_key = f"{prefix}/{key}/{suffix}"
        if add_dist_rank and dist.is_initialized():
            rank = dist.get_rank()
            new_key = f"{new_key}-rank:{rank}"
        new_metrics_dict[new_key] = value
    return new_metrics_dict


def format_per_class_metrics(
    metrics: Mapping[str, Tensor],
    num_classes: int,
    suffix: str = "per_class",
    idx_to_classname: Mapping[int, str] | None = None,
) -> Mapping[str, Tensor]:
    """Format per-class metrics by adding a suffix to each class metric."""
    new_metrics_dict: dict[str, Tensor] = {}
    for key, value in metrics.items():
        for i in range(num_classes):
            if idx_to_classname is not None:
                class_name = idx_to_classname.get(i, i)
                new_key = f"{suffix}/{key}_{class_name}"
            else:
                new_key = f"{suffix}/{key}_class_{i}"
            new_metrics_dict[new_key] = value[i]
    return new_metrics_dict
