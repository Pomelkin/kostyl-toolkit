from typing import TypedDict

from .base import BaseScheduler
from .composite import CompositeScheduler
from .cosine import CosineParamScheduler
from .cosine import CosineScheduler
from .linear import LinearParamScheduler
from .linear import LinearScheduler
from .plateau import PlateauWithAnnealingParamScheduler
from .plateau import PlateauWithAnnealingScheduler


class SchedulerMapping(TypedDict):
    """Map names to scheduler classes."""

    linear: type[LinearScheduler]
    cosine: type[CosineScheduler]
    plateau: type[PlateauWithAnnealingScheduler]
    composite: type[CompositeScheduler]


class ParamSchedulerMapping(TypedDict):
    """Map names to scheduler classes."""

    linear: type[LinearParamScheduler]
    cosine: type[CosineParamScheduler]
    plateau: type[PlateauWithAnnealingParamScheduler]


SCHEDULER_MAPPING: SchedulerMapping = {
    "linear": LinearScheduler,
    "cosine": CosineScheduler,
    "plateau": PlateauWithAnnealingScheduler,
    "composite": CompositeScheduler,
}


PARAM_SCHEDULER_MAPPING: ParamSchedulerMapping = {
    "linear": LinearParamScheduler,
    "cosine": CosineParamScheduler,
    "plateau": PlateauWithAnnealingParamScheduler,
}


__all__ = [
    "PARAM_SCHEDULER_MAPPING",
    "SCHEDULER_MAPPING",
    "BaseScheduler",
    "CompositeScheduler",
    "CosineParamScheduler",
    "CosineScheduler",
    "LinearParamScheduler",
    "LinearScheduler",
    "PlateauWithAnnealingParamScheduler",
    "PlateauWithAnnealingScheduler",
]
