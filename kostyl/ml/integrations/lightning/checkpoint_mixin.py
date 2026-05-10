from pathlib import Path
from typing import Any
from typing import TypeVar
from typing import cast

import torch
from transformers import PretrainedConfig
from transformers import PreTrainedModel

from kostyl.utils.logging import setup_logger


logger = setup_logger("LightningPretrainedModelMixin", fmt="only_message")

TModel = TypeVar("TModel", bound=PreTrainedModel)


class LightningCheckpointLoaderMixin:
    """A mixin class for loading pretrained models from PyTorch Lightning checkpoints."""

    @classmethod
    def from_lightning_checkpoint(  # noqa: C901
        cls: type[TModel],
        checkpoint_path: str | Path,
        config_key: str = "config",
        weights_prefix: str | None = "model.",
        strict_prefix: bool = False,
        **kwargs: Any,
    ) -> TModel:
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
        from_pretrained_kwargs = {
            "proxies": kwargs.pop("proxies", None),
            "output_loading_info": kwargs.pop("output_loading_info", False),
            "_from_pipeline": kwargs.pop("_from_pipeline", None),
            "_from_auto": kwargs.pop("_from_auto", False),
            "dtype": kwargs.pop("dtype", None),
            "torch_dtype": kwargs.pop("torch_dtype", None),
            "device_map": kwargs.pop("device_map", None),
            "max_memory": kwargs.pop("max_memory", None),
            "offload_folder": kwargs.pop("offload_folder", None),
            "offload_buffers": kwargs.pop("offload_buffers", False),
            "quantization_config": kwargs.pop("quantization_config", None),
            "subfolder": kwargs.pop("subfolder", ""),
            "_commit_hash": kwargs.pop("_commit_hash", None),
            "variant": kwargs.pop("variant", None),
            "adapter_kwargs": (kwargs.pop("adapter_kwargs", {}) or {}).copy(),
            "adapter_name": kwargs.pop("adapter_name", "default"),
            "generation_config": kwargs.pop("generation_config", None),
            "gguf_file": kwargs.pop("gguf_file", None),
            "tp_plan": kwargs.pop("tp_plan", None),
            "tp_size": kwargs.pop("tp_size", None),
            "distributed_config": kwargs.pop("distributed_config", None),
            "device_mesh": kwargs.pop("device_mesh", None),
            "trust_remote_code": kwargs.pop("trust_remote_code", None),
            "use_kernels": kwargs.pop("use_kernels", False),
            "kernel_config": kwargs.pop("kernel_config", None),
            "key_mapping": kwargs.pop("key_mapping", None),
            "attn_implementation": kwargs.pop("attn_implementation", None),
        }

        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        if weights_prefix is None:
            weights_prefix = ""

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

        # Load config
        config_cls = cast(type[PretrainedConfig], cls.config_class)
        config_dict = checkpoint_dict[config_key]
        config_dict.update(kwargs)
        config = config_cls.from_dict(config_dict)

        raw_state_dict: dict[str, torch.Tensor] = checkpoint_dict["state_dict"]

        # Handle weights prefix
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

        from_pretrained_kwargs.update(kwargs)

        # Instantiate model and load state dict
        model = cls.from_pretrained(
            pretrained_model_name_or_path=None,
            config=config,
            state_dict=state_dict,
            **from_pretrained_kwargs,  # type: ignore
        )

        return model
