import threading
from pathlib import Path

from filelock import FileLock
from filelock import Timeout


class DirLock:
    """
    Directory-level lock combining per-process and cross-process synchronization.

    The lock protects a directory with a shared in-memory reentrant lock for
    threads in the current process and a `.dirlock` file lock for coordination
    with other processes.
    """

    _registry: dict[str, threading.RLock] = {}  # noqa: RUF012
    _registry_guard = threading.Lock()

    def __init__(
        self,
        directory: str | Path,
        timeout: float = -1,
    ) -> None:
        """
        Initialize a lock for an existing directory.

        Args:
            directory: Directory path to protect.
            timeout: Default timeout in seconds for acquiring the lock. Negative
                values wait indefinitely.

        Raises:
            NotADirectoryError: If `directory` does not point to an existing directory.

        """
        path = Path(directory).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"{path} is not a directory")

        self._dir = path
        self._timeout = timeout

        key = str(path)
        with self._registry_guard:
            self._thread_lock = self._registry.setdefault(key, threading.RLock())

        self._file_lock = FileLock(str(path / ".dirlock"), timeout=timeout)
        return

    def acquire(self, timeout: float | None = None) -> None:
        """
        Acquire the directory lock.

        Args:
            timeout: Timeout in seconds for this acquisition attempt. If None,
                the instance default timeout is used.

        Raises:
            filelock.Timeout: If the thread lock or file lock cannot be acquired
                before the timeout expires.

        """
        t = self._timeout if timeout is None else timeout

        thread_timeout = t if t >= 0 else -1
        if not self._thread_lock.acquire(timeout=thread_timeout):
            raise Timeout(str(self._dir))

        try:
            self._file_lock.acquire(timeout=t)
        except BaseException:
            # If file lock acquisition fails, release the thread lock to avoid deadlock
            self._thread_lock.release()
            raise
        return

    def release(self) -> None:
        """Release the directory lock."""
        # Release in reverse order to avoid deadlocks: file lock first, then thread lock
        try:
            self._file_lock.release()
        finally:
            self._thread_lock.release()
        return

    def __enter__(self) -> "DirLock":
        self.acquire()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.release()
        return

    @property
    def is_locked(self) -> bool:
        """Return whether this instance currently holds the file lock."""
        return self._file_lock.is_locked
