from pathlib import Path

from caseconverter import pascalcase
from caseconverter import snakecase
from clearml import Task

from kostyl.utils.dict_manipulations import convert_to_flat_dict
from kostyl.utils.dict_manipulations import flattened_dict_to_nested
from kostyl.utils.fs import load_config


class ConfigSyncingClearmlMixin[TConfig]:
    """Mixin providing ClearML task configuration syncing functionality for Pydantic models."""

    @classmethod
    def connect_as_file(
        cls: type[TConfig],  # pyright: ignore
        task: Task,
        path: str | Path,
        alias: str | None = None,
    ) -> TConfig:
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
        try:
            model = cls.from_file(path=connected_path_str)
        except AttributeError as e:
            raise AttributeError(
                f"{cls.__name__} must implement from_file method to use. "
                "Inherit from ConfigLoadingMixin to get this functionality or implement your own from_file method."
            ) from e
        return model

    @classmethod
    def connect_as_dict(
        cls: type[TConfig],  # pyright: ignore
        task: Task,
        path: str | Path,
        alias: str | None = None,
    ) -> TConfig:
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
        try:
            model = cls.from_dict(state_dict=config)
        except AttributeError as e:
            raise AttributeError(
                f"{cls.__name__} must implement from_dict method to use. "
                "Inherit from ConfigLoadingMixin to get this functionality or implement your own from_dict method."
            ) from e
        return model
