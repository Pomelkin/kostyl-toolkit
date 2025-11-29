from pathlib import Path
from typing import Any
from typing import cast

import torch
from transformers import PretrainedConfig
from transformers import PreTrainedModel


try:
    from peft import PeftConfig
except ImportError:
    PeftConfig = None  # ty: ignore

from kostyl.utils.logging import log_incompatible_keys
from kostyl.utils.logging import setup_logger


logger = setup_logger("LightningPretrainedModelMixin", fmt="only_message")


class LightningCheckpointLoaderMixin(PreTrainedModel):
    """A mixin class for loading pretrained models from PyTorch Lightning checkpoints."""

    @classmethod
    def from_lighting_checkpoint[TModelInstance: LightningCheckpointLoaderMixin](  # noqa: C901
        cls: type[TModelInstance],
        checkpoint_path: str | Path,
        config_key: str = "config",
        weights_prefix: str = "model.",
        should_log_incompatible_keys: bool = True,
        **kwargs: Any,
    ) -> TModelInstance:
        """
        Load a model from a Lightning checkpoint file.

        This class method loads a pretrained model from a PyTorch Lightning checkpoint file (.ckpt).
        It extracts the model configuration from the checkpoint, instantiates the model, and loads
        the state dictionary, handling any incompatible keys.

        Note:
            The method uses `torch.load` with `weights_only=False` and `mmap=True` for loading.
            Incompatible keys (missing, unexpected, mismatched) are collected and optionally logged.

        Args:
            cls (type["LightningPretrainedModelMixin"]): The class of the model to instantiate.
            checkpoint_path (str | Path): Path to the checkpoint file. Must be a .ckpt file.
            config_key (str, optional): Key in the checkpoint dictionary where the config is stored.
                Defaults to "config".
            weights_prefix (str, optional): Prefix to strip from state dict keys. Defaults to "model.".
                If not empty and doesn't end with ".", a "." is appended.
            should_log_incompatible_keys (bool, optional): Whether to log incompatible keys. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the model loading method.

        Returns:
            TModelInstance: The loaded model instance.

        Raises:
            ValueError: If checkpoint_path is a directory, not a .ckpt file, or invalid.
            FileNotFoundError: If the checkpoint file does not exist.

        """
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_dir():
            raise ValueError(f"{checkpoint_path} is a directory")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{checkpoint_path} does not exist")
        if not checkpoint_path.suffix == ".ckpt":
            raise ValueError(f"{checkpoint_path} is not a .ckpt file")

        checkpoint_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )

        config_cls = cast(PretrainedConfig, type(cls.config_class))
        config_dict = checkpoint_dict[config_key]
        config_dict.update(kwargs)
        config = config_cls.from_dict(config_dict)

        kwargs_for_model = {}
        for key in kwargs:
            if not hasattr(config, key):
                kwargs_for_model[key] = kwargs[key]

        with torch.device("meta"):
            model = cls(config, **kwargs_for_model)

            if "peft_config" in checkpoint_dict:
                if PeftConfig is None:
                    raise ImportError(
                        "peft is not installed. Please install it to load PEFT models."
                    )
                for name, adapter_dict in checkpoint_dict["peft_config"].items():
                    peft_cfg = PeftConfig.from_peft_type(**adapter_dict)
                    model.add_adapter(peft_cfg, adapter_name=name)

        incompatible_keys: dict[str, list[str]] = {}
        if weights_prefix != "":
            if weights_prefix[-1] != ".":
                weights_prefix += "."
            model_state_dict = {}
            mismatched_keys = []
            for key, value in checkpoint_dict["state_dict"].items():
                if key.startswith(weights_prefix):
                    new_key = key[len(weights_prefix) :]
                    model_state_dict[new_key] = value
                else:
                    mismatched_keys.append(key)
                incompatible_keys["mismatched_keys"] = mismatched_keys
        else:
            model_state_dict = checkpoint_dict["state_dict"]

        missing_keys, unexpected_keys = model.load_state_dict(
            model_state_dict, strict=False, assign=True
        )
        incompatible_keys["missing_keys"] = missing_keys
        incompatible_keys["unexpected_keys"] = unexpected_keys
        if should_log_incompatible_keys:
            log_incompatible_keys(incompatible_keys=incompatible_keys, logger=logger)
        return model
