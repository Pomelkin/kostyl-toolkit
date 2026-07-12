from argparse import Namespace
from collections.abc import Callable
from typing import Any
from typing import Literal
from typing import cast
from typing import override

from clearml import Logger as ClearMLSDKLogger
from clearml import OutputModel
from clearml import Task
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.fabric.utilities.logger import _add_prefix
from lightning.fabric.utilities.logger import _convert_params
from lightning.fabric.utilities.logger import _flatten_dict
from lightning.fabric.utilities.logger import _sanitize_params as _utils_sanitize_params
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch import Tensor
from transformers.configuration_utils import PreTrainedConfig


class ClearMLLogger(Logger):
    """
    Log to `ClearML <https://clear.ml>`_ directly via its native ``Logger``.

    The ClearML ``Task`` is created/configured by the caller and passed in. This
    logger does not own its lifecycle (it never closes/marks the task), it only
    reports scalars, connects hyperparameters, and optionally tracks checkpoints
    through an ``OutputModel``.

    Metric keys follow TensorBoard-style grouping: the prefix before ``/`` is the
    group (chart title) and the last segment is the curve (series), e.g.
    ``train/loss`` -> title ``train``, series ``loss``.

    Example::

        from clearml import Task

        task = Task.init(project_name="my_proj", task_name="run-1")
        logger = ClearMLLogger(task=task, prefix="")
        logger.log_hyperparams({"epochs": 5, "optimizer": "Adam"})
        logger.log_metrics({"train/loss": 0.42, "train/f1": 0.81}, step=10)
        logger.finalize("success")
    """

    LOGGER_JOIN_CHAR = "/"
    DEFAULT_GROUP_TITLE = "Scalars"

    def __init__(
        self,
        task: Task,
        prefix: str = "",
        auto_group_scalars: bool = False,
        output_model: OutputModel | None = None,
        upload_checkpoints: bool = True,
        model_config_provider: Callable[[], PreTrainedConfig] | None = None,
        upload_strategy: Literal["best", "last"] | None = None,
    ) -> None:
        """
        Init a ClearMLLogger.

        Args:
            task: An already-initialized ClearML `Task`. The caller is
                responsible for creating it (`Task.init(...)` /
                `Task.get_task(...)`) and for closing/completing it afterwards.
            prefix: A string to put at the beginning of all metric keys, joined
                with `LOGGER_JOIN_CHAR` (`"/"`) so grouping semantics are preserved.
            auto_group_scalars: Mirrors ClearML's `sdk.development.tensorboard.auto_group_scalars` setting.
                When `False` (ClearML default), a
                metric without a `/` is reported under `title == series ==key`.
                When `True`, such metrics are grouped under a single `"Scalars"` title.
            output_model: Optional ClearML `OutputModel` to track checkpoints.
                If given, `after_save_checkpoint` updates its weights on
                every checkpoint save. If `None`, checkpoint hooks are a no-op.
            upload_checkpoints: Only relevant when `output_model` is set. When
                `True` the checkpoint file is uploaded to the model's storage
                (`update_weights(weights_filename=...)`, with
                `auto_delete_file=False` so Lightning keeps the local file).
                When `False` only the checkpoint path/URI is registered without
                copying any bytes (`update_weights(register_uri=...)`).
            model_config_provider: Optional zero-argument callable returning the
                model's `PreTrainedConfig`. When set, `after_save_checkpoint`
                calls it and stores the config diff on the `output_model`
                (`update_design(config_dict=config.to_diff_dict())`) alongside
                the tracked checkpoint. If `None`, no config is attached.
            upload_strategy: Which checkpoint `after_save_checkpoint` tracks:
                `"best"` uses the checkpoint callback's best model path,
                `"last"` uses its last model path. Required (must not be `None`)
                whenever `output_model` is provided; ignored otherwise.

        """
        if output_model is not None and upload_strategy is None:
            raise ValueError(
                "If output_model is provided, then upload_strategy must be specified."
            )

        super().__init__()
        self._task = task
        self._prefix = prefix
        self._auto_group_scalars = auto_group_scalars
        self._output_model = output_model
        self._upload_checkpoints = upload_checkpoints
        self._logger: ClearMLSDKLogger | None = None
        self._logged_checkpoint_path: str | None = None
        self._model_config_provider = model_config_provider
        self._upload_strategy = upload_strategy
        return

    @property
    @override
    def name(self) -> str:
        """ClearML task name."""
        return self._task.name

    @property
    @override
    def version(self) -> str:
        """ClearML task id (globally unique)."""
        return self._task.id

    @property
    def task_id(self) -> str:
        """Alias of :attr:`version` — the ClearML task id."""
        return self.version

    @property
    def task(self) -> Task:
        """The underlying ClearML ``Task``."""
        return self._task

    @property
    def output_model(self) -> OutputModel | None:
        """The ClearML ``OutputModel`` tracking checkpoints, if any."""
        return self._output_model

    @property
    @rank_zero_experiment
    def experiment(self) -> ClearMLSDKLogger:
        """
        The underlying ClearML ``Logger`` (``task.get_logger()``).

        Use it for anything beyond scalars::

            logger.experiment.report_histogram(...)
            logger.experiment.report_image(...)
        """
        if self._logger is None:
            self._logger = self._task.get_logger()
        return self._logger

    @property
    def clearml_sdk_logger(self) -> ClearMLSDKLogger:
        """
        The inner ClearML ``Logger`` **without** rank-zero gating.

        Unlike :attr:`experiment`, returns the real logger on every rank, so
        non-zero ranks can report directly (e.g. per-rank validation images).
        Requires all ranks to be attached to the same ClearML task; the caller
        owns rank discipline and deduplication.
        """
        if self._logger is None:
            self._logger = self._task.get_logger()
        return self._logger

    def _split_tag(self, tag: str) -> tuple[str, str]:
        # num_split_parts=1, split_char="/", join_char="_"
        parts = tag.split("/")
        series = parts[-1]
        title = "_".join(parts[:-1])
        if len(title) == 0:
            title = self.DEFAULT_GROUP_TITLE if self._auto_group_scalars else tag
        return title, series

    # ------------------------------------------------------------------ #
    # logging
    # ------------------------------------------------------------------ #
    @override
    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if rank_zero_only.rank != 0:  # ty:ignore[unresolved-attribute]
            raise RuntimeError("experiment tried to log from global_rank != 0")

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)  # ty:ignore[invalid-assignment]
        iteration = step if step is not None else 0

        for k, v in metrics.items():
            if isinstance(v, Tensor):
                v = v.item()

            if isinstance(v, dict):
                # add_scalars-style: one title, many series
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, Tensor):
                        sub_v = sub_v.item()
                    self.experiment.report_scalar(
                        title=k, series=sub_k, value=sub_v, iteration=iteration
                    )
            else:
                title, series = self._split_tag(k)
                self.experiment.report_scalar(
                    title=title, series=series, value=v, iteration=iteration
                )

    @override
    @rank_zero_only
    def log_hyperparams(
        self,
        params: dict[str, Any] | Namespace,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        params = self._sanitize_params(params)
        # connect is two-way; here it surfaces the dict under the task's
        # Hyperparameters in the ClearML UI.
        self._task.connect(params, name="hparams", ignore_remote_overrides=True)
        return

    @override
    @rank_zero_only
    def save(self) -> None:
        self.experiment.flush()
        return

    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        # Task lifecycle is owned by the caller: flush only, do not close/mark.
        self.experiment.flush()
        return

    @override
    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        """
        Track the saved checkpoint via the ``OutputModel`` (if provided).

        Picks the best checkpoint when available, otherwise the last one. Skips
        if the same path was already logged (avoids re-uploading an unchanged
        ``best`` on every save).
        """
        if self._output_model is None:
            return

        upload_strategy = cast(Literal["best", "last"], self._upload_strategy)

        if upload_strategy == "best" and checkpoint_callback.best_model_path:
            ckpt_path = checkpoint_callback.best_model_path
        elif upload_strategy == "last" and checkpoint_callback.last_model_path:
            ckpt_path = checkpoint_callback.last_model_path
        else:
            return

        if ckpt_path == self._logged_checkpoint_path:
            return

        if self._upload_checkpoints:
            # upload the file; auto_delete_file=False so Lightning keeps the .ckpt
            self._output_model.update_weights(
                weights_filename=ckpt_path, auto_delete_file=False, async_enable=False
            )
        else:
            # register the path/URI only, no bytes copied
            self._output_model.update_weights(register_uri=ckpt_path)

        if self._model_config_provider is not None:
            config = self._model_config_provider()
            diff_dict = config.to_diff_dict()
            self._output_model.update_design(config_dict=diff_dict)

        self._logged_checkpoint_path = ckpt_path
        return

    @staticmethod
    def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
        params = _utils_sanitize_params(params)
        # arrays with dim > 1 are not loggable as params -> stringify
        return {
            k: str(v) if hasattr(v, "ndim") and v.ndim > 1 else v
            for k, v in params.items()
        }
