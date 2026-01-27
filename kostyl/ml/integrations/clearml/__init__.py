try:
    import clearml  # noqa: F401
except ImportError as e:
    raise ImportError(
        "ClearML integration requires the 'clearml' package. "
        "Please install it via 'pip install clearml'."
    ) from e
