from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import override

from clearml import OutputModel
from clearml import Task

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
        task: Task,
        output_model_name: str,
        output_model_tags: list[str] | None = None,
        verbose: bool = True,
        enable_tag_versioning: bool = True,
        label_enumeration: dict[str, int] | None = None,
        config_dict: dict[str, str] | None = None,
    ) -> None:
        """
        Initializes the ClearMLRegistryUploaderCallback.

        Args:
            task: ClearML task.
            ckpt_callback: ModelCheckpoint instance used by Trainer.
            output_model_name: Name for the ClearML output model.
            output_model_tags: Tags for the output model.
            verbose: Whether to log messages.
            label_enumeration: Optional mapping of label names to integer IDs.
            config_dict: Optional configuration dictionary to associate with the model.
            enable_tag_versioning: Whether to enable versioning in tags. If True,
                the version tag (e.g., "v1.0") will be automatically incremented or if not present, added as "v1.0".

        """
        super().__init__()
        if output_model_tags is None:
            output_model_tags = []

        self.task = task
        self.output_model_name = output_model_name
        self.output_model_tags = output_model_tags
        self.config_dict = config_dict
        self.label_enumeration = label_enumeration
        self.verbose = verbose
        self.enable_tag_versioning = enable_tag_versioning

        self.best_model_path: str = ""

        self._output_model: OutputModel | None = None
        self._last_uploaded_model_path: str = ""
        self._upload_callback: Callable | None = None
        return

    def _create_output_model(self) -> OutputModel:
        if self.enable_tag_versioning:
            version = find_version_in_tags(self.output_model_tags)
            if version is None:
                self.output_model_tags.append("v1.0")
            else:
                new_version = increment_version(version)
                self.output_model_tags.remove(version)
                self.output_model_tags.append(new_version)

        if "LightningCheckpoint" not in self.output_model_tags:
            self.output_model_tags.append("LightningCheckpoint")

        return OutputModel(
            task=self.task,
            name=self.output_model_name,
            framework="PyTorch",
            tags=self.output_model_tags,
            config_dict=None,
            label_enumeration=self.label_enumeration,
        )

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

        if self._output_model is None:
            self._output_model = self._create_output_model()

        if self.verbose:
            logger.info(f"Uploading model from {path}")

        self._output_model.update_weights(
            path,
            auto_delete_file=False,
            async_enable=False,
        )
        self._output_model.update_design(config_dict=self.config_dict)

        self._last_uploaded_model_path = path
        return
