import os
import pwd
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING
from typing import override

import rich_click as click

from kostyl.utils.logging import setup_logger


if TYPE_CHECKING:
    import torch


logger = setup_logger(name="hold-gpus", fmt="detailed")

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(b|[kmgt](?:i?b)?)", re.IGNORECASE)
_UNIT_BYTES = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
_MIB = 1024**2


class ByteSize(click.ParamType):
    """Click parameter type for sizes like ``100mb``, ``1.5gb`` or ``1024kb`` (binary units)."""

    name = "size"

    @override
    def convert(
        self, value: str | int, param: click.Parameter | None, ctx: click.Context | None
    ) -> int:
        if isinstance(value, int):
            return value
        match = _SIZE_RE.fullmatch(value.strip())
        if match is None:
            self.fail(
                f"{value!r} is not a size, expected a number with a unit: 100mb, 1.5gb, 1024kb",
                param,
                ctx,
            )
        number, unit = match.groups()
        size = int(float(number) * _UNIT_BYTES[unit[0].lower()])
        if size <= 0:
            self.fail(f"{value!r} must be a positive size", param, ctx)
        return size


@dataclass(frozen=True)
class GpuInfo:
    """A physical GPU as reported by ``nvidia-smi``."""

    index: int
    uuid: str
    name: str


@dataclass(frozen=True)
class GpuProcess:
    """A process holding memory on a GPU."""

    gpu_uuid: str
    pid: int
    used_mib: int | None
    name: str


def format_size(n_bytes: float) -> str:
    """Format a byte count with binary units, e.g. ``1.5 GiB``."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n_bytes < 1024:
            return f"{n_bytes:.4g} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.4g} TiB"


def parse_gpu_ids(values: tuple[str, ...]) -> list[int]:
    """Parse GPU indices given as separate values and/or comma-separated lists, keeping order."""
    ids: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if not part.isdigit():
                raise click.BadParameter(f"{part!r} is not a GPU index")
            if int(part) not in ids:
                ids.append(int(part))
    if not ids:
        raise click.BadParameter("no GPU indices given")
    return ids


def _query_nvidia_smi(target: str, fields: tuple[str, ...]) -> list[list[str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise click.ClickException("nvidia-smi is not found in PATH")
    result = subprocess.run(  # noqa: S603
        [
            executable,
            f"--query-{target}={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    # The last field may be a process name, which can itself contain commas.
    return [
        [value.strip() for value in line.split(",", len(fields) - 1)]
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def list_gpus() -> list[GpuInfo]:
    """List the GPUs visible to ``nvidia-smi``."""
    rows = _query_nvidia_smi("gpu", ("index", "uuid", "name"))
    return [
        GpuInfo(index=int(index), uuid=uuid, name=name) for index, uuid, name in rows
    ]


def list_gpu_processes() -> list[GpuProcess]:
    """List compute processes on all GPUs."""
    rows = _query_nvidia_smi(
        "compute-apps", ("gpu_uuid", "pid", "used_memory", "process_name")
    )
    return [
        GpuProcess(
            gpu_uuid=uuid,
            pid=int(pid),
            used_mib=int(used) if used.isdigit() else None,
            name=Path(name.split(" ", 1)[0]).name,
        )
        for uuid, pid, used, name in rows
    ]


def _user_name(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def _pid_owner(pid: int) -> str:
    try:
        uid = Path(f"/proc/{pid}").stat().st_uid
    except OSError:
        # The process has already exited, lives in another PID namespace, or /proc is restricted.
        return "?"
    return _user_name(uid)


class GpuProcessWatcher:
    """Log processes that appear on or disappear from the watched GPUs, together with their owners."""

    def __init__(self, gpus: list[GpuInfo]) -> None:
        """
        Initialize the watcher.

        Args:
            gpus: GPUs to watch.

        """
        self._index_by_uuid = {gpu.uuid: gpu.index for gpu in gpus}
        self._owners: dict[tuple[str, int], str] = {}
        self._known: dict[tuple[str, int], GpuProcess] = {}
        self._me = _user_name(os.getuid())
        return

    def poll(self, appeared_verb: str = "took") -> None:
        """
        Compare the current GPU processes with the previous poll and log the difference.

        Args:
            appeared_verb: Verb used for processes seen for the first time.

        """
        try:
            processes = list_gpu_processes()
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning(f"Failed to query GPU processes: {exc}")
            return

        current = {
            (proc.gpu_uuid, proc.pid): proc
            for proc in processes
            if proc.gpu_uuid in self._index_by_uuid and proc.pid != os.getpid()
        }
        for key in sorted(current.keys() - self._known.keys(), key=self._sort_key):
            self._owners[key] = _pid_owner(key[1])
            proc = current[key]
            used = (
                "memory" if proc.used_mib is None else format_size(proc.used_mib * _MIB)
            )
            level = "INFO" if self._owners[key] == self._me else "WARNING"
            logger.log(level, f"{self._describe(key, proc)} {appeared_verb} {used}")
        for key in sorted(self._known.keys() - current.keys(), key=self._sort_key):
            logger.info(f"{self._describe(key, self._known[key])} released its memory")
            del self._owners[key]
        self._known = current
        return

    def _sort_key(self, key: tuple[str, int]) -> tuple[int, int]:
        return self._index_by_uuid[key[0]], key[1]

    def _describe(self, key: tuple[str, int], proc: GpuProcess) -> str:
        return f"GPU {self._index_by_uuid[key[0]]}: {self._owners[key]} (pid {proc.pid}, {proc.name})"


def _resolve_gpus(gpu_ids: list[int]) -> list[GpuInfo]:
    try:
        available = {gpu.index: gpu for gpu in list_gpus()}
    except (subprocess.SubprocessError, OSError) as exc:
        raise click.ClickException(
            f"Failed to list GPUs with nvidia-smi: {exc}"
        ) from exc
    missing = [index for index in gpu_ids if index not in available]
    if missing:
        raise click.BadParameter(
            f"no GPU with index {', '.join(map(str, missing))}, available: "
            f"{', '.join(map(str, sorted(available)))}",
            param_hint="GPU_IDS",
        )
    return [available[index] for index in gpu_ids]


def _restrict_cuda_to(gpus: list[GpuInfo]) -> None:
    # Select GPUs by UUID so CUDA ordinals follow the requested order regardless of
    # CUDA_DEVICE_ORDER (CUDA enumerates fastest-first, nvidia-smi by PCI bus).
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    if previous is not None:
        logger.warning(
            f"Ignoring CUDA_VISIBLE_DEVICES={previous!r}: GPU ids are nvidia-smi indices"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu.uuid for gpu in gpus)
    return


def _allocate(gpus: list[GpuInfo], size: int) -> list["torch.Tensor"]:
    import torch

    if torch.cuda.device_count() != len(gpus):
        raise click.ClickException(
            f"CUDA sees {torch.cuda.device_count()} of the {len(gpus)} requested GPUs"
        )
    buffers: list[torch.Tensor] = []
    for ordinal, gpu in enumerate(gpus):
        try:
            # zeros (not empty) touches the memory, so it is committed right away.
            buffers.append(
                torch.zeros(size, dtype=torch.uint8, device=f"cuda:{ordinal}")
            )
        except RuntimeError as exc:
            reason = str(exc).splitlines()[0]
            logger.error(
                f"GPU {gpu.index} ({gpu.name}): failed to allocate {format_size(size)}: {reason}"
            )
            continue
        logger.success(f"GPU {gpu.index} ({gpu.name}): holding {format_size(size)}")
    return buffers


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "gpu_ids",
    nargs=-1,
    required=True,
    callback=lambda ctx, param, value: parse_gpu_ids(value),
)
@click.option(
    "-s",
    "--size",
    type=ByteSize(),
    default="100mb",
    show_default=True,
    help="Memory to hold on each GPU, e.g. 100mb, 1.5gb, 1024kb (binary units: 1kb = 1024 bytes).",
)
@click.option(
    "-i",
    "--interval",
    type=click.FloatRange(min=0.1),
    default=2.0,
    show_default=True,
    help="Seconds between checks for other processes on the held GPUs.",
)
def main(gpu_ids: list[int], size: int, interval: float) -> None:
    """
    Hold memory on GPU_IDS until Ctrl+C and log who else starts using them.

    GPU_IDS are nvidia-smi indices, separated by spaces or commas: `0 1 3` or `0,1,3`.
    Actual usage per GPU is SIZE plus the CUDA context (a few hundred MiB).
    """
    gpus = _resolve_gpus(gpu_ids)
    _restrict_cuda_to(gpus)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    watcher = GpuProcessWatcher(gpus)
    try:
        watcher.poll(appeared_verb="already uses")
        buffers = _allocate(gpus, size)
        if not buffers:
            raise click.ClickException(
                "Could not allocate memory on any of the requested GPUs"
            )
        logger.info(f"Holding {len(buffers)}/{len(gpus)} GPUs, press Ctrl+C to release")
        while True:
            time.sleep(interval)
            watcher.poll()
    except KeyboardInterrupt:
        logger.info("Stopped, GPU memory is released")
    return


if __name__ == "__main__":
    main()
