from typing import Literal
from typing import override

from clearml import OutputModel
from clearml import Task
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint

from kostyl.ml.clearml.logging_utils import find_version_in_tags
from kostyl.ml.clearml.logging_utils import increment_version
from kostyl.ml.lightning import KostylLightningModule
from kostyl.utils.logging import setup_logger


logger = setup_logger()


class ClearMLRegistryUploaderCallback(Callback):
    """PyTorch Lightning callback to upload the best model checkpoint to ClearML."""

    def __init__(
        self,
        task: Task,
        ckpt_callback: ModelCheckpoint,
        output_model_name: str,
        output_model_tags: list[str] | None = None,
        verbose: bool = True,
        enable_tag_versioning: bool = True,
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
        self.ckpt_callback = ckpt_callback
        self.output_model_name = output_model_name
        self.output_model_tags = output_model_tags
        self.verbose = verbose
        self.uploading_frequency = uploading_frequency
        self.enable_tag_versioning = enable_tag_versioning

        self._output_model: OutputModel | None = None
        self._last_best_model_path: str = ""
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
        config = pl_module.model_config
        if config is not None:
            config = config.to_dict()

        return OutputModel(
            task=self.task,
            name=self.output_model_name,
            framework="PyTorch",
            tags=self.output_model_tags,
            config_dict=config,
        )

    def _upload_best_checkpoint(self, pl_module: "KostylLightningModule") -> None:
        current_best = self.ckpt_callback.best_model_path

        if not current_best:
            if self.verbose:
                logger.info("No best model found yet to upload")
            return

        if current_best == self._last_best_model_path:
            if self.verbose:
                logger.info("Best model unchanged since last upload")
            return

        if self._output_model is None:
            self._output_model = self._create_output_model(pl_module)

        if self.verbose:
            logger.info(f"Uploading best model from {current_best}")

        self._output_model.update_weights(
            current_best,
            auto_delete_file=False,
            async_enable=False,
        )

        self._last_best_model_path = current_best
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
