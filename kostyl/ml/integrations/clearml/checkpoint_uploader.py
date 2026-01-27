from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import override

from clearml import OutputModel

from kostyl.ml.base_uploader import ModelCheckpointUploader
from kostyl.utils.logging import setup_logger


logger = setup_logger()


class ClearMLCheckpointUploader(ModelCheckpointUploader):
    """PyTorch Lightning callback to upload the best model checkpoint to ClearML."""

    def __init__(
        self,
        model_name: str,
        config_dict: dict[str, str] | None = None,
        label_enumeration: dict[str, int] | None = None,
        tags: list[str] | None = None,
        comment: str | None = None,
        framework: str | None = None,
        base_model_id: str | None = None,
        new_model_per_upload: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the ClearMLRegistryUploader.

        Args:
            model_name: The name for the newly created model.
            label_enumeration: The label enumeration dictionary of string (label) to integer (value) pairs.
            config_dict: Optional configuration dictionary to associate with the model.
            tags: A list of strings which are tags for the model.
            comment: A comment / description for the model.
            framework: The framework of the model (e.g., "PyTorch", "TensorFlow").
            base_model_id: Optional ClearML model ID to use as a base for the new model
            new_model_per_upload: Whether to create a new ClearML model
                for every upload or update weights of the same model. When updating weights,
                the last uploaded checkpoint will be replaced (and deleted).
            verbose: Whether to log messages during upload.

        """
        super().__init__()
        if base_model_id is not None and new_model_per_upload:
            raise ValueError(
                "Cannot set base_model_id when new_model_per_upload is True."
            )

        self.verbose = verbose
        self.new_model_per_upload = new_model_per_upload
        self.best_model_path: str = ""
        self.config_dict = config_dict
        self._output_model: OutputModel | None = None
        self._last_uploaded_model_path: str = ""
        self._upload_callback: Callable | None = None

        self._validate_tags(tags)
        self.model_fabric = partial(
            OutputModel,
            name=model_name,
            label_enumeration=label_enumeration,
            tags=tags,
            comment=comment,
            framework=framework,
            base_model_id=base_model_id,
        )
        return

    @staticmethod
    def _validate_tags(tags: list[str] | None) -> None:
        if tags is None:
            return
        if "LightningCheckpoint" not in tags:
            tags.append("LightningCheckpoint")
        return None

    @property
    def output_model_(self) -> OutputModel:
        """Returns the OutputModel instance based on `new_model_per_upload` setting."""
        if self.new_model_per_upload:
            model = self.model_fabric()
            self._output_model = self.model_fabric()
        else:
            if self._output_model is None:
                self._output_model = self.model_fabric()
            model = self._output_model
        return model

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

        self.output_model_.update_weights(
            path,
            auto_delete_file=False,
            async_enable=False,
        )
        self.output_model_.update_design(config_dict=self.config_dict)

        self._last_uploaded_model_path = path
        return
