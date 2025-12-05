from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Literal
from typing import override

from clearml import OutputModel
from clearml import Task
from lightning import Trainer
from lightning.pytorch.callbacks import Callback

from kostyl.ml.clearml.logging_utils import find_version_in_tags
from kostyl.ml.clearml.logging_utils import increment_version
from kostyl.ml.lightning import KostylLightningModule
from kostyl.utils.logging import setup_logger


logger = setup_logger()


class RegistryUploaderCallback(Callback, ABC):
    """Abstract Lightning callback responsible for tracking and uploading the best-performing model checkpoint."""

    @property
    @abstractmethod
    def best_model_path(self) -> str:
        """Return the file system path pointing to the best model artifact produced during training."""
        raise NotImplementedError

    @best_model_path.setter
    @abstractmethod
    def best_model_path(self, value: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def _upload_best_checkpoint(self, pl_module: "KostylLightningModule") -> None:
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
        uploading_frequency: Literal[
            "after-every-eval", "on-train-end"
        ] = "on-train-end",
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
            uploading_frequency: When to upload:
                - "after-every-eval": after each validation phase.
                - "on-train-end": once at the end of training.

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
        self.uploading_frequency = uploading_frequency
        self.enable_tag_versioning = enable_tag_versioning

        self._output_model: OutputModel | None = None
        self._last_uploaded_model_path: str = ""
        self._best_model_path: str = ""
        self._upload_callback: Callable | None = None
        return

    @property
    @override
    def best_model_path(self) -> str:
        return self._best_model_path

    @best_model_path.setter
    @override
    def best_model_path(self, value: str) -> None:
        self._best_model_path = value
        if self._upload_callback is not None:
            self._upload_callback()
        return

    def _create_output_model(self, pl_module: "KostylLightningModule") -> OutputModel:
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

        if self.config_dict is None:
            config = pl_module.model_config
            if config is not None:
                config = config.to_dict()
        else:
            config = self.config_dict

        return OutputModel(
            task=self.task,
            name=self.output_model_name,
            framework="PyTorch",
            tags=self.output_model_tags,
            config_dict=None,
            label_enumeration=self.label_enumeration,
        )

    @override
    def _upload_best_checkpoint(self, pl_module: "KostylLightningModule") -> None:
        if not self._best_model_path or (
            self._best_model_path == self._last_uploaded_model_path
        ):
            if not self._best_model_path:
                if self.verbose:
                    logger.info("No best model found yet to upload")
            elif self._best_model_path == self._last_uploaded_model_path:
                if self.verbose:
                    logger.info("Best model unchanged since last upload")
            self._upload_callback = partial(self._upload_best_checkpoint, pl_module)
            return
        self._upload_callback = None

        if self._output_model is None:
            self._output_model = self._create_output_model(pl_module)

        if self.verbose:
            logger.info(f"Uploading best model from {self._best_model_path}")

        self._output_model.update_weights(
            self._best_model_path,
            auto_delete_file=False,
            async_enable=False,
        )
        if self.config_dict is None:
            config = pl_module.model_config
            if config is not None:
                config = config.to_dict()
        else:
            config = self.config_dict
        self._output_model.update_design(config_dict=config)

        self._last_uploaded_model_path = self._best_model_path
        return

    @override
    def on_validation_end(
        self, trainer: Trainer, pl_module: "KostylLightningModule"
    ) -> None:
        if self.uploading_frequency != "after-every-eval":
            return
        if not trainer.is_global_zero:
            return

        self._upload_best_checkpoint(pl_module)
        return

    @override
    def on_train_end(
        self, trainer: Trainer, pl_module: "KostylLightningModule"
    ) -> None:
        if not trainer.is_global_zero:
            return

        self._upload_best_checkpoint(pl_module)
        return
