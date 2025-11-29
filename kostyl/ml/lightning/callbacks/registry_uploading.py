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
        uploading_frequency: Literal[
            "after-every-eval", "on-train-end"
        ] = "on-train-end",
    ) -> None:
        """
        Initialize the callback.

        Args:
            task (Task): The ClearML task object.
            ckpt_callback (ModelCheckpoint): The model checkpoint callback.
            output_model_name (str): The name for the output model.
            output_model_tags (list[str] | None, optional): Tags for the output model. Defaults to None, which is converted to an empty list.
            verbose (bool, optional): Whether to log verbose messages. Defaults to True.
            uploading_frequency (Literal["after-every-eval", "on-train-end"]): Frequency of uploading the model. Defaults to "on-train-end".

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

        self._output_model: OutputModel | None = None
        self._last_best_model_path: str = ""
        return

    def _create_output_model(self, pl_module: KostylLightningModule) -> OutputModel:
        version = find_version_in_tags(self.output_model_tags)
        if version is None:
            self.output_model_tags.append("v1.0")
        else:
            new_version = increment_version(version)
            self.output_model_tags.remove(version)
            self.output_model_tags.append(new_version)

        config = pl_module.model_config
        if config is not None:
            config = config.to_dict()

        output_model = OutputModel(
            task=self.task,
            name=self.output_model_name,
            framework="PyTorch",
            tags=self.output_model_tags,
            config_dict=config,
        )
        return output_model

    @override
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: KostylLightningModule
    ) -> None:
        if (not trainer.is_global_zero) or (
            self.uploading_frequency != "after-every-eval"
        ):
            return
        if self._output_model is None:
            self._output_model = self._create_output_model(pl_module)

        if self.ckpt_callback.best_model_path != self._last_best_model_path:
            if self.verbose:
                logger.info(f"Best model path: {self.ckpt_callback.best_model_path}")
            self._output_model.update_weights(
                self.ckpt_callback.best_model_path,
                auto_delete_file=False,
                async_enable=True,
            )
            self._last_best_model_path = self.ckpt_callback.best_model_path
        elif self.verbose:
            logger.info("Best model path: unchanged since last upload, skipping...")
        return

    @override
    def on_train_end(self, trainer: Trainer, pl_module: KostylLightningModule) -> None:
        if not trainer.is_global_zero:
            return
        if self._output_model is None:
            self._output_model = self._create_output_model(pl_module)

        if self.ckpt_callback.best_model_path != self._last_best_model_path:
            if self.verbose:
                logger.info(
                    f"Best model path at the end of training: {self.ckpt_callback.best_model_path}"
                )
            self._output_model.update_weights(
                self.ckpt_callback.best_model_path,
                auto_delete_file=False,
                async_enable=False,
            )
        elif self.verbose:
            logger.info(
                "Best model path at the end of training: unchanged since last upload, skipping..."
            )
        return
