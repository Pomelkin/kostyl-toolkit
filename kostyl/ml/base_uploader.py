from abc import ABC
from abc import abstractmethod
from pathlib import Path

from kostyl.utils.logging import setup_logger


logger = setup_logger()


class ModelCheckpointUploader(ABC):
    """Abstract base class for uploading model checkpoints to a registry backend."""

    @abstractmethod
    def upload_checkpoint(self, path: str | Path) -> None:
        """Upload the checkpoint located at the given path to the configured registry backend."""
        raise NotImplementedError
