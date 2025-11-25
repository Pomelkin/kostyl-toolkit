from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel as PydanticBaseModel

from kostyl.utils.fs import load_config


TConfig = TypeVar("TConfig", bound=PydanticBaseModel)


class ConfigLoadingMixin:
    """Pydantic mixin class providing basic configuration loading functionality."""

    @classmethod
    def from_file(
        cls: type[TConfig],  # pyright: ignore
        path: str | Path,
    ) -> TConfig:
        """
        Create an instance of the class from a configuration file.

        Args:
            cls_: The class type to instantiate.
            path (str | Path): Path to the configuration file.

        Returns:
            An instance of the class created from the configuration file.

        """
        config = load_config(path)
        instance = cls.model_validate(config)
        return instance

    @classmethod
    def from_dict(
        cls: type[TConfig],  # pyright: ignore
        state_dict: dict,
    ) -> TConfig:
        """
        Creates an instance from a dictionary.

        Args:
            cls_: The class type to instantiate.
            state_dict (dict): A dictionary representing the state of the
                class that must be validated and used for initialization.

        Returns:
            An initialized instance of the class based on the
                provided state dictionary.

        """
        instance = cls.model_validate(state_dict)
        return instance
