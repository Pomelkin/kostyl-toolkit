from pathlib import Path

import yaml


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
