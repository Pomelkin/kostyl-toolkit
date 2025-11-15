from typing import override

from clearml import OutputModel
from clearml import Task
from lightning import LightningModule
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint

from teremok.ml_core.clearml.logging_utils import find_version_in_tags
from teremok.ml_core.clearml.logging_utils import increment_version
from teremok.utils.logging import setup_logger


logger = setup_logger("callbacks/clearml.py")


class ClearMLRegistryUploaderCallback(Callback):
    """PyTorch Lightning callback to upload the best model checkpoint to ClearML."""

    def __init__(
        self,
        task: Task,
        ckpt_callback: ModelCheckpoint,
        output_model_name: str,
        output_model_tags: list[str] | None = None,
    ) -> None:
        """
        Initialize the callback.

        Args:
            task (Task): The ClearML task object.
            ckpt_callback (ModelCheckpoint): The model checkpoint callback.
            output_model_name (str): The name for the output model.
            output_model_tags (list[str] | None, optional): Tags for the output model. Defaults to None, which is converted to an empty list.

        """
        super().__init__()
        if output_model_tags is None:
            output_model_tags = []
        self.task = task
        self.ckpt_callback = ckpt_callback
        self.output_model_name = output_model_name
        self.output_model_tags = output_model_tags
        return

    @override
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        version = find_version_in_tags(self.output_model_tags)
        if version is None:
            self.output_model_tags.append("v1.0")
        else:
            new_version = increment_version(version)
            self.output_model_tags.remove(version)
            self.output_model_tags.append(new_version)

        output_model = OutputModel(
            task=self.task,
            name=self.output_model_name,
            framework="PyTorch",
            tags=self.output_model_tags,
            config_dict=pl_module.get_model_config(return_dict=True),  # type: ignore
        )
        if self.ckpt_callback.best_model_path != "":
            logger.info(f"Uploading best model: {self.ckpt_callback.best_model_path}")
            output_model.update_weights(
                self.ckpt_callback.best_model_path,
                auto_delete_file=False,
                async_enable=False,
            )
        return
