import pytest
import rich_click as click

from kostyl.ml.cli.hold_gpus import ByteSize
from kostyl.ml.cli.hold_gpus import format_size
from kostyl.ml.cli.hold_gpus import parse_gpu_ids


class TestByteSize:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("100mb", 100 * 1024**2),
            ("100MB", 100 * 1024**2),
            ("100MiB", 100 * 1024**2),
            ("100m", 100 * 1024**2),
            ("1gb", 1024**3),
            ("1.5 gb", 1536 * 1024**2),
            ("1024kb", 1024**2),
            ("512b", 512),
        ],
    )
    def test_valid(self, value: str, expected: int) -> None:
        assert ByteSize().convert(value, None, None) == expected

    @pytest.mark.parametrize("value", ["100", "mb", "-1mb", "0mb", "1pb", "1,5gb"])
    def test_invalid(self, value: str) -> None:
        with pytest.raises(click.BadParameter):
            ByteSize().convert(value, None, None)


class TestParseGpuIds:
    def test_spaces_and_commas(self) -> None:
        assert parse_gpu_ids(("0", "3,1", "2,")) == [0, 3, 1, 2]

    def test_duplicates_are_dropped(self) -> None:
        assert parse_gpu_ids(("1,1", "1")) == [1]

    @pytest.mark.parametrize("values", [(), (",",), ("a",), ("-1",)])
    def test_invalid(self, values: tuple[str, ...]) -> None:
        with pytest.raises(click.BadParameter):
            parse_gpu_ids(values)


def test_format_size() -> None:
    assert format_size(100 * 1024**2) == "100 MiB"
    assert format_size(1536 * 1024**2) == "1.5 GiB"
    assert format_size(512) == "512 B"
