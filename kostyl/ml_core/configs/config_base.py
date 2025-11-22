from pathlib import Path
from typing import TypeVar

import clearml
import yaml
from caseconverter import pascalcase
from caseconverter import snakecase
from pydantic import BaseModel as PydanticBaseModel

from kostyl.utils import convert_to_flat_dict
from kostyl.utils import flattened_dict_to_nested


def load_config(path: Path | str) -> dict:
    """Load a configuration from file."""
    if isinstance(path, str):
        path = Path(path)

    if not path.is_file():
        raise ValueError(f"Config file {path} does not exist or is not a file.")

    match path.suffix:
        case ".yaml" | ".yml":
            config = yaml.safe_load(path.open("r"))
        case _:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    return config


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


TModel = TypeVar("TModel", bound="ClearMLBaseModel")


class ClearMLConfigMixin(ConfigLoadingMixin):
    """Pydantic mixin class providing ClearML configuration loading and syncing functionality."""

    @classmethod
    def connect_as_file(
        cls: type[TModel],  # pyright: ignore
        task: clearml.Task,
        path: str | Path,
        alias: str | None = None,
    ) -> TModel:
        """
        Connects the configuration file to a ClearML task and creates an instance of the class from it.

        This method connects the specified configuration file to the given ClearML task for version control and monitoring,
        then loads and validates the configuration to the class.

        Args:
            cls: The class type to instantiate.
            task: The ClearML Task object to connect the configuration to.
            path: Path to the configuration file (supports YAML format).
            alias: Optional alias for the configuration in ClearML. Defaults to PascalCase of the class name if None.

        Returns:
            An instance of the class created from the connected configuration file.

        """
        if isinstance(path, Path):
            str_path = str(path)
        else:
            str_path = path

        name = alias if alias is not None else pascalcase(cls.__name__)
        connected_path = task.connect_configuration(str_path, name=pascalcase(name))

        if not isinstance(connected_path, str):
            connected_path_str = str(connected_path)
        else:
            connected_path_str = connected_path

        model = cls.from_file(path=connected_path_str)
        return model

    @classmethod
    def connect_as_dict(
        cls: type[TModel],  # pyright: ignore
        task: clearml.Task,
        path: str | Path,
        alias: str | None = None,
    ) -> TModel:
        """
        Connects configuration from a file as a dictionary to a ClearML task and creates an instance of the class.

        This class method loads configuration from a file as a dictionary, flattens and sync them with ClearML
        task parameters. Then it creates an instance of the class using the synced dictionary.

        Args:
            cls: The class type of the model to be created (must be a TRetuningModel subclass).
            task: The ClearML task to connect the configuration to.
            path: Path to the configuration file to load parameters from.
            alias: Optional alias name for the configuration. If None, uses snake_case of class name.

        Returns:
            An instance of the specified class created from the loaded configuration.

        """
        name = alias if alias is not None else snakecase(cls.__name__)

        config = load_config(path)

        flattened_config = convert_to_flat_dict(config)
        task.connect(flattened_config, name=pascalcase(name))
        config = flattened_dict_to_nested(flattened_config)

        model = cls.from_dict(state_dict=config)
        return model


class ClearMLBaseModel(PydanticBaseModel, ClearMLConfigMixin):
    """A Pydantic model class with ClearML configuration loading and syncing functionality."""

    pass
