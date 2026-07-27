from pathlib import Path
from typing import Any


try:
    from kostyl.ml.dist_utils import local_rank_zero_only
except ImportError:

    def local_rank_zero_only(func: Any) -> Any:
        """Dummy decorator for non-distributed environments."""
        return func


def convert_to_flat_dict(
    nested_dict: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """
    Converts a nested dictionary to a flat one.

    Example: {'a': 1, 'b': {'c': 2}} -> {'a': 1, 'b.c': 2}
    """
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(convert_to_flat_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flattened_dict_to_nested(
    flattened_dict: dict[str, Any], sep: str = "."
) -> dict[str, Any]:
    """
    Converts a flat dictionary back to a nested one.

    Example: {'a': 1, 'b.c': 2} -> {'a': 1, 'b': {'c': 2}}
    """
    result = {}
    for key, value in flattened_dict.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]

        d[parts[-1]] = value
    return result


def load_file(path: Path | str) -> dict:
    """Load a file from the specified path."""
    if isinstance(path, str):
        path = Path(path)

    if not path.is_file():
        raise ValueError(f"File {path} does not exist or is not a file.")

    match path.suffix:
        case ".yaml" | ".yml":
            import yaml

            config = yaml.safe_load(path.open("r"))
        case ".json":
            import orjson

            config = orjson.loads(path.read_bytes())
        case _:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    return config


@local_rank_zero_only
def dump_to_file(data: dict[Any, Any] | list[Any], path: Path | str) -> None:
    """
    Dump data into a file at the specified path.

    The file format is determined by the file extension (supports .yaml/.yml and .json).
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.parent.exists():
        raise ValueError(f"Directory {path.parent} does not exist.")

    match path.suffix:
        case ".yaml" | ".yml":
            import yaml

            with path.open("w") as f:
                yaml.safe_dump(data, f)
        case ".json":
            import orjson

            path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        case _:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    return


# alias for backward compatibility, will be removed in future versions
dump_into_file = dump_to_file


def is_empty_dir(path: Path) -> bool:
    """Check if a directory is empty or not."""
    return path.is_dir() and not any(path.iterdir())


def is_overridden(cls: type, method_name: str) -> bool:
    """Check if a method is overridden in a subclass."""
    return method_name in cls.__dict__
