# Kostyl Toolkit

Kickass Orchestration System for Training, Yielding & Logging — a batteries-included toolbox that glues PyTorch Lightning, Hugging Face Transformers, and ClearML into a single workflow.

## Overview
- Rapidly bootstrap Lightning experiments with opinionated defaults (`KostylLightningModule`, custom schedulers, grad clipping and metric formatting).
- Keep model configs source-controlled via Pydantic mixins, with ClearML syncing out of the box (`ConfigLoadingMixin`, `ClearMLConfigMixin`).
- Reuse Lightning checkpoints directly inside Transformers models through `LightningCheckpointLoaderMixin`.
- Ship distributed-friendly utilities (deterministic logging, FSDP helpers, LR scaling, ClearML tag management).

## Installation
```bash
# Latest release from PyPI
pip install kostyl-toolkit

# or with uv
uv pip install kostyl-toolkit
```

Development setup:
```bash
uv sync                # creates the virtualenv declared in pyproject.toml
source .venv/bin/activate.fish
pre-commit install     # optional but recommended
```

## Quick Start
```python
from lightning import Trainer
from transformers import AutoModelForSequenceClassification

from kostyl.ml_core.configs.hyperparams import HyperparamsConfig
from kostyl.ml_core.configs.training_params import TrainingParams
from kostyl.ml_core.lightning.extenstions.custom_module import KostylLightningModule


class TextClassifier(KostylLightningModule):
	def __init__(self, hyperparams: HyperparamsConfig):
				super().__init__()
		self.hyperparams = hyperparams  # grad clipping + scheduler knobs
				self.model = AutoModelForSequenceClassification.from_pretrained(
						"distilbert-base-uncased",
						num_labels=2,
				)

		def training_step(self, batch, batch_idx):
				outputs = self.model(**batch)
				self.log("train/loss", outputs.loss)
				return outputs.loss

train_cfg = TrainingParams.from_file("configs/training.yaml")
hyperparams = HyperparamsConfig.from_file("configs/hyperparams.yaml")

module = TextClassifier(hyperparams)

trainer = Trainer(**train_cfg.trainer.model_dump())
trainer.fit(module)
```

Restoring a plain Transformers model from a Lightning checkpoint:
```python
from kostyl.ml_core.lightning.extenstions.pretrained_model import LightningCheckpointLoaderMixin


model = LightningCheckpointLoaderMixin.from_lighting_checkpoint(
		"checkpoints/epoch=03-step=500.ckpt",
		config_key="config",
		weights_prefix="model.",
)
```

## Components
- **Configurations** (`kostyl/ml_core/configs`): strongly-typed training, optimizer, and scheduler configs with ClearML syncing helpers.
- **Lightning Extensions** (`kostyl/ml_core/lightning`): custom LightningModule base class, callbacks, logging bridges, and the checkpoint loader mixin.
- **Schedulers** (`kostyl/ml_core/schedulers`): extensible LR schedulers (base/composite/cosine) with serialization helpers and on-step logging.
- **ClearML Utilities** (`kostyl/ml_core/clearml`): tag/version helpers and logging bridges for ClearML Tasks.
- **Distributed + Metrics Utils** (`kostyl/ml_core/dist_utils.py`, `metrics_formatting.py`): world-size-aware LR scaling, rank-aware metric naming, and per-class formatting.
- **Logging Helpers** (`kostyl/utils/logging.py`): rank-aware Loguru setup and uniform handling of incompatible checkpoint keys.

## Project Layout
```
kostyl/
	ml_core/
		configs/                # Pydantic configs + ClearML mixins
		lightning/              # Lightning module, callbacks, loggers, extensions
		schedulers/             # Base + composite/cosine schedulers
		clearml/                # Logging + pulling utilities
	utils/                    # Dict helpers, logging utilities
```
