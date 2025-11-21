import re

from kostyl.utils import setup_logger


logger = setup_logger(name="clearml_logging_utils.py", fmt="only_message")


def increment_version(s: str) -> str:
    """
    Increments the minor part of a version string.

    Examples:
        v1.00 -> v1.01
        v2.99 -> v2.100
        v.3.009 -> v.3.010

    """
    s = s.strip()
    m = re.fullmatch(r"v(\.?)(\d+)\.(\d+)", s)
    if not m:
        raise ValueError(f"Invalid version format: {s!r}. Expected 'vX.Y' or 'v.X.Y'.")

    vdot, major_str, minor_str = m.groups()
    major = int(major_str)
    minor = int(minor_str) + 1

    # preserve leading zeros based on original width, length may increase (99 -> 100)
    minor_out = str(minor).zfill(len(minor_str))
    prefix = f"v{vdot}"  # 'v' or 'v.'
    return f"{prefix}{major}.{minor_out}"


def find_version_in_tags(tags: list[str]) -> str | None:
    """
    Finds the first version tag in the list of tags.

    Note:
        Version tags must be in the format 'vX.Y' or 'v.X.Y' (an optional dot after 'v' is supported).

    """
    version_pattern = re.compile(r"^v(\.?)(\d+)\.(\d+)$")
    for tag in tags:
        if version_pattern.match(tag):
            return tag
    return None
