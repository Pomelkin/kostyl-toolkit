from pathlib import Path
from typing import Any
from typing import cast

import torch
from transformers import PretrainedConfig
from transformers import PreTrainedModel

from kostyl.utils.logging import setup_logger


logger = setup_logger("LightningPretrainedModelMixin", fmt="only_message")


class LightningCheckpointLoaderMixin(PreTrainedModel):
    """A mixin class for loading pretrained models from PyTorch Lightning checkpoints."""

    @classmethod
    def from_lightning_checkpoint[TModelInstance: LightningCheckpointLoaderMixin](  # noqa: C901
        cls: type[TModelInstance],
        checkpoint_path: str | Path,
        config_key: str = "config",
        weights_prefix: str | None = "model.",
        strict_prefix: bool = False,
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
            weights_prefix (str | None, optional): Prefix to strip from state dict keys. Defaults to "model.".
                If not empty and doesn't end with ".", a "." is appended. If empty or None, no prefix stripping will be skipped.
            strict_prefix (bool, optional): If True, drop tensors those keys that do not start with the
                specified prefix. Defaults to False.
            kwargs: Additional keyword arguments to pass to the model's `from_pretrained` method.

        Returns:
            TModelInstance: The loaded model instance.

        Raises:
            ValueError: If checkpoint_path is a directory, not a .ckpt file, or invalid.
            FileNotFoundError: If the checkpoint file does not exist.

        """
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        if weights_prefix is None:
            weights_prefix = ""
        weights_prefix = cast(str, weights_prefix)
        if weights_prefix == "" and strict_prefix:
            logger.warning(
                "strict_prefix=True has no effect when weights_prefix is empty or None."
            )

        if checkpoint_path.is_dir():
            raise ValueError(f"{checkpoint_path} is a directory")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{checkpoint_path} does not exist")
        if checkpoint_path.suffix != ".ckpt":
            raise ValueError(f"{checkpoint_path} is not a .ckpt file")

        checkpoint_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )

        # 1. Восстанавливаем конфиг
        config_cls = cast(type[PretrainedConfig], cls.config_class)
        config_dict = checkpoint_dict[config_key]
        config_dict.update(kwargs)
        config = config_cls.from_dict(config_dict)

        kwargs_for_model: dict[str, Any] = {}
        for key, value in kwargs.items():
            if not hasattr(config, key):
                kwargs_for_model[key] = value

        raw_state_dict: dict[str, torch.Tensor] = checkpoint_dict["state_dict"]

        if weights_prefix:
            if not weights_prefix.endswith("."):
                weights_prefix = weights_prefix + "."
            state_dict: dict[str, torch.Tensor] = {}
            matched_keys_counter = 0
            for key, value in raw_state_dict.items():
                if key.startswith(weights_prefix):
                    new_key = key[len(weights_prefix) :]
                    state_dict[new_key] = value
                    matched_keys_counter += 1
                elif not strict_prefix:
                    state_dict[key] = value

            if matched_keys_counter == 0:
                if strict_prefix:
                    raise ValueError(
                        f"No keys in the checkpoint start with the specified prefix '{weights_prefix}'. "
                        "Try to load with `strict_prefix=False` or verify the prefix."
                    )
                else:
                    logger.warning(
                        f"No keys in the checkpoint start with the specified prefix '{weights_prefix}'. "
                    )
        else:
            state_dict = raw_state_dict

        model = cls.from_pretrained(
            pretrained_model_name_or_path=None,
            config=config,
            state_dict=state_dict,
            **kwargs_for_model,
        )

        return model
