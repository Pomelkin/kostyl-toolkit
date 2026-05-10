from pathlib import Path
from typing import Any
from typing import TypeVar
from typing import cast

from clearml import InputModel
from clearml import Task
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import SentencePieceBackend
from transformers import TokenizersBackend


try:
    from kostyl.ml.integrations.lightning import (
        LightningCheckpointLoaderMixin,  # pyright: ignore[reportAssignmentType]
    )

    LIGHTING_MIXIN_AVAILABLE = True
except ImportError:

    class LightningCheckpointLoaderMixin(PreTrainedModel):  # noqa: D101
        @classmethod
        def from_lightning_checkpoint(cls, *args: Any, **kwargs: Any) -> Any:  # noqa: D102
            raise ImportError(
                "Loading from Lightning checkpoints requires lightning integration. "
                "Please package install via 'pip install lightning' to enable this functionality."
            )

    LIGHTING_MIXIN_AVAILABLE = False


def load_tokenizer_from_clearml(
    tokenizer_id: str,
    task: Task | None = None,
    ignore_remote_overrides: bool = True,
    name: str | None = None,
) -> tuple[SentencePieceBackend | TokenizersBackend, InputModel]:
    """
    Retrieve a Hugging Face tokenizer stored in a ClearML.

    Args:
        tokenizer_id (str): The ClearML InputModel identifier that holds the tokenizer artifacts.
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
    clearml_tokenizer = InputModel(model_id=tokenizer_id)
    if task is not None:
        clearml_tokenizer.connect(
            task, ignore_remote_overrides=ignore_remote_overrides, name=name
        )

    tokenizer = AutoTokenizer.from_pretrained(
        clearml_tokenizer.get_local_copy(raise_on_error=True)
    )
    if tokenizer is None:
        raise ValueError(
            f"Failed to load tokenizer from ClearML InputModel with id: {tokenizer_id}"
        )
    return tokenizer, clearml_tokenizer


TModel = TypeVar(
    "TModel", bound=PreTrainedModel | LightningCheckpointLoaderMixin | AutoModel
)


def load_model_from_clearml(
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
    clearml_model = InputModel(model_id=model_id)

    if task is not None:
        clearml_model.connect(
            task,
            ignore_remote_overrides=ignore_remote_overrides,
            name=name,
        )

    local_path = Path(clearml_model.get_local_copy(raise_on_error=True))

    if local_path.is_dir() and clearml_model._is_package():
        if not issubclass(model, (PreTrainedModel, AutoModel)):
            raise ValueError(
                f"Model class {model.__name__} must be a subclass of PreTrainedModel or AutoModel "
                "to be loaded from a ClearML package directory."
            )
        model_instance = model.from_pretrained(local_path, **kwargs)  # type: ignore

    elif local_path.suffix == ".ckpt":
        if not LIGHTING_MIXIN_AVAILABLE:
            raise ImportError(
                "Loading from Lightning checkpoints requires lightning integration. "
                "Please package install via 'pip install lightning' to enable this functionality."
            )
        if not issubclass(model, LightningCheckpointLoaderMixin):
            raise ValueError(
                f"Model class {model.__name__} is not compatible with Lightning checkpoints "
                "(must inherit from LightningCheckpointLoaderMixin)."
            )
        model_instance = model.from_lightning_checkpoint(local_path, **kwargs)  # type: ignore

    else:
        raise ValueError(
            f"Unsupported model format for path: {local_path}. "
            "Expected a ClearML package directory or a .ckpt file."
        )

    model_instance = cast(TModel, model_instance)
    return model_instance, clearml_model
