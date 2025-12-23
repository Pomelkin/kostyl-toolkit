from pathlib import Path
from typing import Any
from typing import cast

from clearml import InputModel
from clearml import Task
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

from kostyl.ml.lightning.extensions.pretrained_model import (
    LightningCheckpointLoaderMixin,
)


def get_tokenizer_from_clearml(
    model_id: str,
    task: Task | None = None,
    ignore_remote_overrides: bool = True,
    name: str | None = None,
) -> tuple[PreTrainedTokenizerBase, InputModel]:
    """
    Retrieve a Hugging Face tokenizer stored in a ClearML.

    Args:
        model_id (str): The ClearML InputModel identifier that holds the tokenizer artifacts.
        task (Task | None, optional): An optional ClearML Task used to associate and sync
            the model. Defaults to None.
        ignore_remote_overrides (bool, optional): Whether to ignore remote hyperparameter
            overrides when connecting the ClearML task. Defaults to True.
        name: The model name to be stored on the Task
            (default to filename of the model weights, without the file extension, or to Input Model if that is not found)

    Returns:
        The instantiated tokenizer loaded from the local copy
            of the referenced ClearML InputModel and the ClearML InputModel instance.

    """
    clearml_tokenizer = InputModel(model_id=model_id)
    if task is not None:
        clearml_tokenizer.connect(
            task, ignore_remote_overrides=ignore_remote_overrides, name=name
        )

    tokenizer = AutoTokenizer.from_pretrained(
        clearml_tokenizer.get_local_copy(raise_on_error=True)
    )
    return tokenizer, clearml_tokenizer


def get_model_from_clearml[
    TModel: PreTrainedModel | LightningCheckpointLoaderMixin | AutoModel
](
    model_id: str,
    model: type[TModel],
    task: Task | None = None,
    ignore_remote_overrides: bool = True,
    name: str | None = None,
    **kwargs: Any,
) -> tuple[TModel, InputModel]:
    """
    Retrieve a pretrained model from ClearML and instantiate it using the appropriate loader.

    Args:
        model_id: Identifier of the ClearML input model to retrieve.
        model: The model class that implements either PreTrainedModel or LightningCheckpointLoaderMixin.
        task: Optional ClearML task used to resolve the input model reference. If provided, the input model
            will be connected to this task, with remote overrides optionally ignored.
        ignore_remote_overrides: When connecting the input model to the provided task,
            determines whether remote configuration overrides should be ignored.
        name: The model name to be stored on the Task
            (default to filename of the model weights, without the file extension, or to Input Model if that is not found)
        **kwargs: Additional keyword arguments to pass to the model loading method.

    Returns:
        The instantiated model and the ClearML InputModel instance.

    """
    input_model = InputModel(model_id=model_id)

    if task is not None:
        input_model.connect(
            task,
            ignore_remote_overrides=ignore_remote_overrides,
            name=name,
        )

    local_path = Path(input_model.get_local_copy(raise_on_error=True))

    if local_path.is_dir() and input_model._is_package():
        model_instance = model.from_pretrained(local_path, **kwargs)
    elif local_path.suffix == ".ckpt":
        if not issubclass(model, LightningCheckpointLoaderMixin):
            raise ValueError(
                f"Model class {model.__name__} is not compatible with Lightning checkpoints."
            )
        model_instance = model.from_lightning_checkpoint(local_path, **kwargs)
    else:
        raise ValueError(
            f"Unsupported model format for path: {local_path}. "
            "Expected a ClearML package directory or a .ckpt file."
        )
    model_instance = cast(TModel, model_instance)
    return model_instance, input_model
