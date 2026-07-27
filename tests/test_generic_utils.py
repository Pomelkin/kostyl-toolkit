from pathlib import Path

import pytest

from kostyl.utils import convert_to_flat_dict
from kostyl.utils import dump_to_file
from kostyl.utils import flattened_dict_to_nested
from kostyl.utils import is_overridden
from kostyl.utils import load_file


class TestDictConversions:
    def test_convert_to_flat_dict(self) -> None:
        nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        assert convert_to_flat_dict(nested) == {"a": 1, "b.c": 2, "b.d.e": 3}

    def test_flattened_dict_to_nested(self) -> None:
        flat = {"a": 1, "b.c": 2, "b.d.e": 3}
        assert flattened_dict_to_nested(flat) == {"a": 1, "b": {"c": 2, "d": {"e": 3}}}

    def test_roundtrip(self) -> None:
        nested = {"trainer": {"max_epochs": 3, "precision": "bf16"}, "seed": 42}
        assert flattened_dict_to_nested(convert_to_flat_dict(nested)) == nested

    def test_custom_separator(self) -> None:
        nested = {"a": {"b": 1}}
        flat = convert_to_flat_dict(nested, sep="/")
        assert flat == {"a/b": 1}
        assert flattened_dict_to_nested(flat, sep="/") == nested


class TestFileIO:
    @pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
    def test_dump_and_load_roundtrip(self, tmp_path: Path, suffix: str) -> None:
        data = {"lr": 0.001, "scheduler": {"type": "cosine"}}
        path = tmp_path / f"config{suffix}"
        dump_to_file(data, path)
        assert load_file(path) == data

    def test_load_file_accepts_str_path(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        dump_to_file({"a": 1}, path)
        assert load_file(str(path)) == {"a": 1}

    def test_load_file_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            load_file(tmp_path / "missing.yaml")

    def test_load_file_unsupported_format(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_file(path)

    def test_dump_into_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            dump_to_file({"a": 1}, tmp_path / "missing" / "config.yaml")


class TestIsOverridden:
    def test_detects_override(self) -> None:
        class Base:
            def method(self) -> None: ...

        class Child(Base):
            def method(self) -> None: ...

        class UntouchedChild(Base):
            pass

        assert is_overridden(Child, "method")
        assert not is_overridden(UntouchedChild, "method")
