from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import override

from clearml import OutputModel

from kostyl.ml.clearml.logging_utils import find_version_in_tags
from kostyl.ml.clearml.logging_utils import increment_version
from kostyl.utils.logging import setup_logger


logger = setup_logger()


class RegistryUploaderCallback(ABC):
    """Abstract Lightning callback responsible for tracking and uploading the best-performing model checkpoint."""

    @abstractmethod
    def upload_checkpoint(self, path: str | Path) -> None:
        """Upload the checkpoint located at the given path to the configured registry backend."""
        raise NotImplementedError


class ClearMLRegistryUploaderCallback(RegistryUploaderCallback):
    """PyTorch Lightning callback to upload the best model checkpoint to ClearML."""

    def __init__(
        self,
        output_model: OutputModel,
        config_dict: dict[str, str] | None = None,
        verbose: bool = True,
        enable_tag_versioning: bool = False,
    ) -> None:
        """
        Initializes the ClearMLRegistryUploaderCallback.

        Args:
            output_model: ClearML OutputModel instance representing the model to upload.
            verbose: Whether to log messages during upload.
            config_dict: Optional configuration dictionary to associate with the model.
            enable_tag_versioning: Whether to enable versioning in tags. If True,
                the version tag (e.g., "v1.0") will be automatically incremented or if not present, added as "v1.0".

        """
        super().__init__()
        self.output_model = output_model
        self.config_dict = config_dict
        self.verbose = verbose
        self.enable_tag_versioning = enable_tag_versioning

        self.best_model_path: str = ""

        self._last_uploaded_model_path: str = ""
        self._upload_callback: Callable | None = None

        self._validate_tags()
        return

    def _validate_tags(self) -> None:
        output_model_tags = self.output_model.tags or []
        if self.enable_tag_versioning:
            version = find_version_in_tags(output_model_tags)
            if version is None:
                output_model_tags.append("v1.0")
            else:
                new_version = increment_version(version)
                output_model_tags.remove(version)
                output_model_tags.append(new_version)
        if "LightningCheckpoint" not in output_model_tags:
            output_model_tags.append("LightningCheckpoint")
        self.output_model.tags = output_model_tags
        return None

    @override
    def upload_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        if isinstance(path, Path):
            path = str(path)
        if path == self._last_uploaded_model_path:
            if self.verbose:
                logger.info("Model unchanged since last upload")
            return

        if self.verbose:
            logger.info(f"Uploading model from {path}")

        self.output_model.update_weights(
            path,
            auto_delete_file=False,
            async_enable=False,
        )
        self.output_model.update_design(config_dict=self.config_dict)

        self._last_uploaded_model_path = path
        return
