# Kostyl Toolkit

Utilities for the training code that keeps getting copied between projects: typed
configs, PyTorch Lightning helpers, Hugging Face checkpoint loading, optimizer and
scheduler factories, ClearML glue, and small distributed-training helpers.

The package is intentionally small. Core utilities have only lightweight
dependencies; ML integrations are imported from explicit subpackages and require
their own runtime packages.

## Installation

```bash
pip install kostyl-toolkit
```

For the ML helpers:

```bash
pip install "kostyl-toolkit[ml]"
```

Lightning and ClearML integrations also require their upstream packages:

```bash
pip install lightning clearml
```

For local development:

```bash
uv sync
pre-commit install
```

`uv sync` uses the dependency groups in `pyproject.toml`, including the heavier ML
stack used during development.

## What Is Inside

- `kostyl.utils`: file loading, dict flattening/nesting, override checks, and Loguru setup.
- `kostyl.ml.configs`: Pydantic config structs for trainer settings, data settings, optimizers, LR/weight-decay schedules, and simple config loading mixins.
- `kostyl.ml.optim`: optimizer and scheduler factories built around those config structs.
- `kostyl.ml.integrations.lightning`: a `KostylLightningModule`, checkpoint loading mixins for Transformers models/configs, step estimation, and Lightning-specific helpers.
- `kostyl.ml.integrations.lightning.callbacks`: checkpoint and early-stopping callback builders.
- `kostyl.ml.integrations.clearml`: config syncing, checkpoint upload, tokenizer/model loading, dataset helpers, and tag version helpers for ClearML.
- `kostyl.ml.dist_utils`: rank helpers and learning-rate scaling by distributed world size.
- `kostyl.ml.param_groups`: parameter group creation with common no-decay handling.

## Typed Configs

The built-in structs are plain Pydantic models. Use `model_validate` directly, or
combine them with `ConfigLoadingMixin` when you want `.from_file()` and
`.from_dict()` constructors.

```python
from kostyl.ml.configs import ConfigLoadingMixin
from kostyl.ml.configs import HyperparamsConfig
from kostyl.ml.configs import TrainingSettings


class ExperimentTrainingSettings(ConfigLoadingMixin, TrainingSettings):
    pass


class ExperimentHyperparams(ConfigLoadingMixin, HyperparamsConfig):
    pass


training = ExperimentTrainingSettings.from_file("configs/training.yaml")
hyperparams = ExperimentHyperparams.from_file("configs/hyperparams.yaml")
```

`TrainingSettings` covers Lightning trainer settings, early stopping,
checkpointing, and data paths. `HyperparamsConfig` covers optimizer selection,
learning-rate scheduling, weight-decay scheduling, and gradient clipping.

## Lightning Module Base

`KostylLightningModule` adds a few project conventions on top of
`lightning.LightningModule`:

- saves the underlying model config into Lightning checkpoints
- applies gradient clipping in `on_before_optimizer_step`
- supports stage-prefixed metric logging through `log_dict(..., stage="train")`
- logs scheduler values for schedulers based on `BaseScheduler`

Subclasses are expected to expose the model, optional config, and gradient clipping
value through properties.

```python
from lightning import Trainer
from torch import nn
from transformers import AutoModelForSequenceClassification
from transformers import PretrainedConfig

from kostyl.ml.integrations.lightning import KostylLightningModule


class TextClassifier(KostylLightningModule):
    def __init__(self, grad_clip_val: float | None = None) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,
        )
        self._grad_clip_val = grad_clip_val

    @property
    def model_instance(self) -> nn.Module:
        return self.model

    @property
    def model_config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def grad_clip_val(self) -> float | None:
        return self._grad_clip_val

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log("train/loss", outputs.loss)
        return outputs.loss


trainer = Trainer(max_epochs=3, accelerator="auto")
trainer.fit(TextClassifier(grad_clip_val=1.0), train_dataloaders=...)
```

## Lightning Callbacks

Callback builders live in the callbacks subpackage:

```python
from pathlib import Path

from kostyl.ml.configs import CheckpointConfig
from kostyl.ml.integrations.lightning.callbacks import setup_checkpoint_callback


checkpoint_callback = setup_checkpoint_callback(
    dirpath=Path("checkpoints"),
    ckpt_cfg=CheckpointConfig(monitor="val/loss", mode="min"),
)
```

`setup_checkpoint_callback` can also upload saved checkpoints when given a
`ModelCheckpointUploader`, for example `ClearMLCheckpointUploader`.

## Loading Transformers Models From Lightning Checkpoints

Use `LightningCheckpointModelMixin` when a Transformers model class should be
restored directly from a `.ckpt` file produced by Lightning. The mixin reads
`state_dict`, optionally strips a prefix such as `model.`, and calls the model's
`from_pretrained(..., state_dict=...)`.

```python
from transformers import BertConfig
from transformers import BertForSequenceClassification

from kostyl.ml.integrations.lightning import LightningCheckpointModelMixin


class BertClassifierFromLightning(
    LightningCheckpointModelMixin,
    BertForSequenceClassification,
):
    pass


config = BertConfig(num_labels=2)

model = BertClassifierFromLightning.from_lightning_checkpoint(
    "checkpoints/epoch=03-step=500.ckpt",
    config=config,
    weights_prefix="model.",
)
```

If `config` is not passed, the mixin builds it from the checkpoint:

```python
model = BertClassifierFromLightning.from_lightning_checkpoint(
    "checkpoints/epoch=03-step=500.ckpt",
    config_key="config",
    weights_prefix="model.",
)
```

`config_key` is only used when `config=None`. It names the checkpoint entry that
contains the serialized Hugging Face config.

Use `LightningCheckpointConfigMixin` when you only need the config:

```python
from transformers import BertConfig

from kostyl.ml.integrations.lightning import LightningCheckpointConfigMixin


class BertConfigFromLightning(LightningCheckpointConfigMixin, BertConfig):
    pass


config = BertConfigFromLightning.from_lightning_checkpoint(
    "checkpoints/epoch=03-step=500.ckpt",
    config_key="config",
)
```

Backward-compatible aliases are kept for the older loader names, but new code
should prefer `LightningCheckpointModelMixin` and `LightningCheckpointConfigMixin`.

## Optimizers And Schedulers

Optimizer and scheduler factories consume the config structs from
`kostyl.ml.configs`.

```python
from kostyl.ml.configs import AdamConfig
from kostyl.ml.configs import Lr
from kostyl.ml.optim import create_optimizer
from kostyl.ml.optim import create_scheduler
from kostyl.ml.param_groups import create_param_groups


param_groups = create_param_groups(model, lr=3e-4, weight_decay=0.01)

optimizer = create_optimizer(
    parameters_groups=param_groups,
    optimizer_config=AdamConfig(type="AdamW"),
    lr=3e-4,
    weight_decay=0.01,
)

lr_scheduler = create_scheduler(
    config=Lr(
        scheduler_type="cosine",
        base_value=3e-4,
        final_value=3e-5,
        warmup_ratio=0.05,
        warmup_value=1e-6,
    ),
    param_group_field="lr",
    num_iters=10_000,
    optim=optimizer,
)
```

Schedulers expose `current_value()` and are designed to be logged from
`KostylLightningModule.log_scheduled_values()`.

## ClearML Integration

ClearML helpers are under `kostyl.ml.integrations.clearml`.

```python
from pathlib import Path

from clearml import Task

from kostyl.ml.integrations.clearml import ClearMLCheckpointUploader
from kostyl.ml.integrations.lightning.callbacks import setup_checkpoint_callback


task = Task.init(project_name="experiments", task_name="train")

uploader = ClearMLCheckpointUploader(
    model_name="bert-classifier",
    tags=["text-classification"],
)

checkpoint_callback = setup_checkpoint_callback(
    dirpath=Path("checkpoints"),
    ckpt_cfg=training.checkpoint,
    checkpoint_uploader=uploader,
    upload_strategy="only-best",
)
```

`load_model_from_clearml` can restore either a packaged Transformers model
directory or a Lightning `.ckpt` when the target class inherits from
`LightningCheckpointModelMixin`.

## Distributed Helpers

```python
from kostyl.ml.dist_utils import get_global_rank
from kostyl.ml.dist_utils import is_local_rank_zero
from kostyl.ml.dist_utils import scale_lrs_by_world_size


if is_local_rank_zero():
    print(f"global rank: {get_global_rank()}")

scaled_lrs = scale_lrs_by_world_size({"model": 3e-4, "head": 1e-3})
```

## Compatibility Notes

- Python `>=3.10` is required.
- Core installation is intentionally lightweight.
- `kostyl.ml.integrations.lightning` requires `lightning`.
- `kostyl.ml.integrations.clearml` requires `clearml`.
- Low-precision Adam variants require `torchao`.
- New checkpoint mixin names are `LightningCheckpointModelMixin` and
  `LightningCheckpointConfigMixin`; old loader aliases are kept for compatibility.

## Project Layout

```text
kostyl/
  ml/
    configs/                 # Pydantic config structs and loading mixins
    dist_utils/              # rank helpers, LR scaling, FSDP helpers
    integrations/
      clearml/               # ClearML syncing, loading, upload, version helpers
      lightning/             # Lightning module, callbacks, checkpoint mixins
    optim/                   # optimizer and scheduler factories
    param_groups.py          # parameter group builder
  utils/                     # generic helpers and logging setup
```
