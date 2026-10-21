from typing import TYPE_CHECKING
from typing import Any


if TYPE_CHECKING:
    from kostyl.utils.pydantic_models import StrictPydanticBaseModel


__all__ = ["StrictPydanticBaseModel"]


def __getattr__(name: str) -> Any:
    # pydantic ships only with the `ml` extra, so import lazily to keep `import kostyl` working without it.
    if name == "StrictPydanticBaseModel":
        from kostyl.utils.pydantic_models import StrictPydanticBaseModel

        return StrictPydanticBaseModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
