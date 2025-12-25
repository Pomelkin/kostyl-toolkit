from copy import deepcopy
from typing import Any

import torch
from transformers import DataCollatorWithPadding
from transformers.data.data_collator import DataCollatorMixin


class BatchCollatorWithKeyAlignment:
    """
    Maps dataset keys to HuggingFace DataCollator expected keys and collates the batch.

    HuggingFace collators expect specific keys depending on the collator type:
        - `DataCollatorWithPadding`: "input_ids", "attention_mask", "token_type_ids" (optional).
        - `DataCollatorForLanguageModeling`: "input_ids", "attention_mask", "special_tokens_mask" (optional).
        - `DataCollatorForSeq2Seq`: "input_ids", "attention_mask", "labels".
        - `DataCollatorForTokenClassification`: "input_ids", "attention_mask", "labels".

    This wrapper allows you to map arbitrary dataset keys to these expected names before collation,
    optionally truncating sequences to a maximum length.
    """

    def __init__(
        self,
        collator: DataCollatorWithPadding | DataCollatorMixin,
        keys_mapping: dict[str, str] | None = None,
        keys_to_keep: set[str] | None = None,
        max_length: int | None = None,
    ) -> None:
        """
        Initialize the BatchCollatorWithKeyAlignment.

        Args:
        collator: A callable (usually a Hugging Face DataCollator) that takes a list
            of dictionaries and returns a collated batch (e.g., padded tensors).
        keys_mapping: A dictionary mapping original keys to new keys.
        keys_to_keep: A set of keys to retain as-is from the original items.
        max_length: If provided, truncates "input_ids" and "attention_mask" to this length.

        Raises:
            ValueError: If both `keys_mapping` and `keys_to_keep` are None.

        """
        if (keys_mapping is None) and (keys_to_keep is None):
            raise ValueError("Either keys_mapping or keys_to_keep must be provided.")

        if keys_mapping is None:
            keys_mapping = {}
        if keys_to_keep is None:
            keys_to_keep = set()

        self.collator = collator
        self.keys_mapping = deepcopy(keys_mapping)
        self.max_length = max_length

        keys_to_keep_mapping = {v: v for v in keys_to_keep}
        self.keys_mapping.update(keys_to_keep_mapping)

    def _truncate_data(self, key: str, value: Any) -> Any:
        match value:
            case torch.Tensor():
                if value.dim() > 2:
                    raise ValueError(
                        f"Expected value with dim <= 2 for key {key}, got {value.dim()}"
                    )
            case list():
                if isinstance(value[0], list):
                    raise ValueError(
                        f"Expected value with dim <= 2 for key {key}, got nested lists"
                    )
        value = value[: self.max_length]
        return value

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Align keys and collate the batch.

        Args:
            batch: A list of dictionaries representing the data batch.

        Returns:
            The collated batch returned by the underlying collator.

        """
        aligned_batch = []
        for item in batch:
            new_item = {}
            for k in item.keys():
                new_key = self.keys_mapping.get(k, None)
                if new_key is None:
                    continue
                value = item[k]
                if self.max_length is not None and new_key in (
                    "input_ids",
                    "attention_mask",
                ):
                    value = self._truncate_data(new_key, value)
                new_item[new_key] = value
            aligned_batch.append(new_item)

        collated_batch = self.collator(aligned_batch)
        return collated_batch
