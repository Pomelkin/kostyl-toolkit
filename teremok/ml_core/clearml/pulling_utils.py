from clearml import InputModel
from clearml import Task
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase


def get_tokenizer_from_clearml(
    model_id: str, task: Task | None = None, ignore_remote_overrides: bool = True
) -> PreTrainedTokenizerBase:
    """
    Retrieve a Hugging Face tokenizer stored in a ClearML.

    Args:
        model_id (str): The ClearML InputModel identifier that holds the tokenizer artifacts.
        task (Task | None, optional): An optional ClearML Task used to associate and sync
            the model. Defaults to None.
        ignore_remote_overrides (bool, optional): Whether to ignore remote hyperparameter
            overrides when connecting the ClearML task. Defaults to True.

    Returns:
        PreTrainedTokenizerBase: The instantiated tokenizer loaded from the local copy
            of the referenced ClearML InputModel.

    """
    clearml_tokenizer = InputModel(model_id=model_id)
    if task is not None:
        clearml_tokenizer.connect(task, ignore_remote_overrides=ignore_remote_overrides)

    tokenizer = AutoTokenizer.from_pretrained(clearml_tokenizer.get_local_copy())
    return tokenizer
