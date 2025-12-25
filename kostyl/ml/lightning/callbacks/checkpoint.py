from datetime import timedelta
from pathlib import Path
from shutil import rmtree
from typing import Literal
from typing import override

import lightning.pytorch as pl
import torch.distributed as dist
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import ModelCheckpoint

from kostyl.ml.configs import CheckpointConfig
from kostyl.ml.dist_utils import is_main_process
from kostyl.ml.lightning import KostylLightningModule
from kostyl.ml.registry_uploader import RegistryUploaderCallback
from kostyl.utils import setup_logger


logger = setup_logger("callbacks/checkpoint.py")


class ModelCheckpointWithRegistryUploader(ModelCheckpoint):
    r"""
    Save the model after every epoch by monitoring a quantity. Every logged metrics are passed to the
    :class:`~lightning.pytorch.loggers.logger.Logger` for the version it gets saved in the same directory as the
    checkpoint.

    After training finishes, use :attr:`best_model_path` to retrieve the path to the
    best checkpoint file and :attr:`best_model_score` to get its score.

    .. note::
        When using manual optimization with ``every_n_train_steps``, you should save the model state
        in your ``training_step`` before the optimizer step if you want the checkpoint to reflect
        the pre-optimization state. Example:

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                # ... forward pass, loss calculation, backward pass ...

                # Save model state before optimization
                if not hasattr(self, 'saved_models'):
                    self.saved_models = {}
                self.saved_models[batch_idx] = {
                    k: v.detach().clone()
                    for k, v in self.layer.state_dict().items()
                }

                # Then perform optimization
                optimizer.zero_grad()
                self.manual_backward(loss)
                optimizer.step()

                # Optional: Clean up old states to save memory
                if batch_idx > 10:  # Keep last 10 states
                    del self.saved_models[batch_idx - 10]

    Args:
        dirpath: Directory to save the model file.
            Example: ``dirpath='my/path/'``.

            .. warning::
                In a distributed environment like DDP, it's recommended to provide a `dirpath` to avoid race conditions.
                When using manual optimization with ``every_n_train_steps``, make sure to save the model state
                in your training loop as shown in the example above.

            Can be remote file paths such as `s3://mybucket/path/` or 'hdfs://path/'
            (default: ``None``). If dirpath is ``None``, we only keep the ``k`` best checkpoints
            in memory, and do not save anything to disk.

        filename: Checkpoint filename. Can contain named formatting options to be auto-filled.
            If no name is provided, it will be ``None`` and the checkpoint will be saved to
            ``{epoch}``.and if the Trainer uses a logger, the path will also contain logger name and version.

        filename: checkpoint filename. Can contain named formatting options to be auto-filled.

            Example::

                # save any arbitrary metrics like `val_loss`, etc. in name
                # saves a file like: my/path/epoch=2-val_loss=0.02-other_metric=0.03.ckpt
                >>> checkpoint_callback = ModelCheckpoint(
                ...     dirpath='my/path',
                ...     filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'
                ... )

            By default, filename is ``None`` and will be set to ``'{epoch}-{step}'``, where "epoch" and "step" match
            the number of finished epoch and optimizer steps respectively.
        monitor: quantity to monitor. By default it is ``None`` which saves a checkpoint only for the last epoch.
        verbose: verbosity mode. Default: ``False``.
        save_last: When ``True``, saves a `last.ckpt` copy whenever a checkpoint file gets saved. Can be set to
            ``'link'`` on a local filesystem to create a symbolic link. This allows accessing the latest checkpoint
            in a deterministic manner. Default: ``None``.
        save_top_k: if ``save_top_k == k``,
            the best k models according to the quantity monitored will be saved.
            If ``save_top_k == 0``, no models are saved.
            If ``save_top_k == -1``, all models are saved.
            Please note that the monitors are checked every ``every_n_epochs`` epochs.
            If ``save_top_k >= 2`` and the callback is called multiple times inside an epoch, and the filename remains
            unchanged, the name of the saved file will be appended with a version count starting with ``v1`` to avoid
            collisions unless ``enable_version_counter`` is set to False. The version counter is unrelated to the top-k
            ranking of the checkpoint, and we recommend formatting the filename to include the monitored metric to avoid
            collisions.
        save_on_exception: Whether to save a checkpoint when an exception is raised. Default: ``False``.
        mode: one of {min, max}.
            If ``save_top_k != 0``, the decision to overwrite the current save file is made
            based on either the maximization or the minimization of the monitored quantity.
            For ``'val_acc'``, this should be ``'max'``, for ``'val_loss'`` this should be ``'min'``, etc.
        auto_insert_metric_name: When ``True``, the checkpoints filenames will contain the metric name.
            For example, ``filename='checkpoint_{epoch:02d}-{acc:02.0f}`` with epoch ``1`` and acc ``1.12`` will resolve
            to ``checkpoint_epoch=01-acc=01.ckpt``. Is useful to set it to ``False`` when metric names contain ``/``
            as this will result in extra folders.
            For example, ``filename='epoch={epoch}-step={step}-val_acc={val/acc:.2f}', auto_insert_metric_name=False``
        save_weights_only: if ``True``, then only the model's weights will be
            saved. Otherwise, the optimizer states, lr-scheduler states, etc are added in the checkpoint too.
        every_n_train_steps: How many training steps to wait before saving a checkpoint. This does not take into account
            the steps of the current epoch. If ``every_n_train_steps == None or every_n_train_steps == 0``,
            no checkpoints
            will be saved during training. Mutually exclusive with ``train_time_interval`` and ``every_n_epochs``.

            .. note::
                When using with manual optimization, the checkpoint will be saved after the optimizer step by default.
                To save the model state before the optimizer step, you need to save the model state in your
                ``training_step`` before calling ``optimizer.step()``. See the class docstring for an example.
        train_time_interval: Checkpoints are monitored at the specified time interval.
            For all practical purposes, this cannot be smaller than the amount
            of time it takes to process a single training batch. This is not
            guaranteed to execute at the exact time specified, but should be close.
            This must be mutually exclusive with ``every_n_train_steps`` and ``every_n_epochs``.
        every_n_epochs: Number of epochs between checkpoints.
            This value must be ``None`` or non-negative.
            To disable saving top-k checkpoints, set ``every_n_epochs = 0``.
            This argument does not impact the saving of ``save_last=True`` checkpoints.
            If all of ``every_n_epochs``, ``every_n_train_steps`` and
            ``train_time_interval`` are ``None``, we save a checkpoint at the end of every epoch
            (equivalent to ``every_n_epochs = 1``).
            If ``every_n_epochs == None`` and either ``every_n_train_steps != None`` or ``train_time_interval != None``,
            saving at the end of each epoch is disabled
            (equivalent to ``every_n_epochs = 0``).
            This must be mutually exclusive with ``every_n_train_steps`` and ``train_time_interval``.
            Setting both ``ModelCheckpoint(..., every_n_epochs=V, save_on_train_epoch_end=False)`` and
            ``Trainer(max_epochs=N, check_val_every_n_epoch=M)``
            will only save checkpoints at epochs 0 < E <= N
            where both values for ``every_n_epochs`` and ``check_val_every_n_epoch`` evenly divide E.
        save_on_train_epoch_end: Whether to run checkpointing at the end of the training epoch.
            If ``True``, checkpoints are saved at the end of every training epoch.
            If ``False``, checkpoints are saved at the end of validation.
            If ``None`` (default), checkpointing behavior is determined based on training configuration.
            If ``val_check_interval`` is a str, dict, or `timedelta` (time-based), checkpointing is performed after
            validation.
            If ``check_val_every_n_epoch != 1``, checkpointing will not be performed at the end of
            every training epoch. If there are no validation batches of data, checkpointing will occur at the
            end of the training epoch. If there is a non-default number of validation runs per training epoch
            (``val_check_interval != 1``), checkpointing is performed after validation.
        enable_version_counter: Whether to append a version to the existing file name.
            If ``False``, then the checkpoint files will be overwritten.

    Note:
        For extra customization, ModelCheckpoint includes the following attributes:

        - ``CHECKPOINT_JOIN_CHAR = "-"``
        - ``CHECKPOINT_EQUALS_CHAR = "="``
        - ``CHECKPOINT_NAME_LAST = "last"``
        - ``FILE_EXTENSION = ".ckpt"``
        - ``STARTING_VERSION = 1``

        For example, you can change the default last checkpoint name by doing
        ``checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"``

        If you want to checkpoint every N hours, every M train batches, and/or every K val epochs,
        then you should create multiple ``ModelCheckpoint`` callbacks.

        If the checkpoint's ``dirpath`` changed from what it was before while resuming the training,
        only ``best_model_path`` will be reloaded and a warning will be issued.

        If you provide a ``filename`` on a mounted device where changing permissions is not allowed (causing ``chmod``
        to raise a ``PermissionError``), install `fsspec>=2025.5.0`. Then the error is caught, the file's permissions
        remain unchanged, and the checkpoint is still saved. Otherwise, no checkpoint will be saved and training stops.

    Raises:
        MisconfigurationException:
            If ``save_top_k`` is smaller than ``-1``,
            if ``monitor`` is ``None`` and ``save_top_k`` is none of ``None``, ``-1``, and ``0``, or
            if ``mode`` is none of ``"min"`` or ``"max"``.
        ValueError:
            If ``trainer.save_checkpoint`` is ``None``.

    Example::

        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import ModelCheckpoint

        # saves checkpoints to 'my/path/' at every epoch
        >>> checkpoint_callback = ModelCheckpoint(dirpath='my/path/')
        >>> trainer = Trainer(callbacks=[checkpoint_callback])

        # save epoch and val_loss in name
        # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        >>> checkpoint_callback = ModelCheckpoint(
        ...     monitor='val_loss',
        ...     dirpath='my/path/',
        ...     filename='sample-mnist-{epoch:02d}-{val_loss:.2f}'
        ... )

        # save epoch and val_loss in name, but specify the formatting yourself (e.g. to avoid problems with Tensorboard
        # or Neptune, due to the presence of characters like '=' or '/')
        # saves a file like: my/path/sample-mnist-epoch02-val_loss0.32.ckpt
        >>> checkpoint_callback = ModelCheckpoint(
        ...     monitor='val/loss',
        ...     dirpath='my/path/',
        ...     filename='sample-mnist-epoch{epoch:02d}-val_loss{val/loss:.2f}',
        ...     auto_insert_metric_name=False
        ... )

        # retrieve the best checkpoint after training
        >>> checkpoint_callback = ModelCheckpoint(dirpath='my/path/')
        >>> trainer = Trainer(callbacks=[checkpoint_callback])
        >>> model = ...  # doctest: +SKIP
        >>> trainer.fit(model)  # doctest: +SKIP
        >>> print(checkpoint_callback.best_model_path)  # doctest: +SKIP

    .. tip:: Saving and restoring multiple checkpoint callbacks at the same time is supported under variation in the
        following arguments:

        *monitor, mode, every_n_train_steps, every_n_epochs, train_time_interval*

        Read more: :ref:`Persisting Callback State <extensions/callbacks_state:save callback state>`

    """  # noqa: D205

    def __init__(  # noqa: D107
        self,
        registry_uploader_callback: RegistryUploaderCallback,
        uploading_mode: Literal["only-best", "every-checkpoint"] = "only-best",
        dirpath: _PATH | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: bool | Literal["link"] | None = None,
        save_top_k: int = 1,
        save_on_exception: bool = False,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        enable_version_counter: bool = True,
    ) -> None:
        self.registry_uploader_callback = registry_uploader_callback
        self.process_group: dist.ProcessGroup | None = None
        self.uploading_mode = uploading_mode
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_on_exception=save_on_exception,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )
        return

    @override
    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule | KostylLightningModule,
        stage: str,
    ) -> None:
        super().setup(trainer, pl_module, stage)
        if isinstance(pl_module, KostylLightningModule):
            self.process_group = pl_module.get_process_group()
        return

    @override
    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        if dist.is_initialized():
            dist.barrier(group=self.process_group)
        if trainer.is_global_zero and self.registry_uploader_callback is not None:
            match self.uploading_mode:
                case "every-checkpoint":
                    self.registry_uploader_callback.upload_checkpoint(filepath)
                case "only-best":
                    if filepath == self.best_model_path:
                        self.registry_uploader_callback.upload_checkpoint(filepath)
        return


def setup_checkpoint_callback(
    dirpath: Path,
    ckpt_cfg: CheckpointConfig,
    save_weights_only: bool = True,
    registry_uploader_callback: RegistryUploaderCallback | None = None,
    uploading_mode: Literal["only-best", "every-checkpoint"] | None = None,
) -> ModelCheckpointWithRegistryUploader | ModelCheckpoint:
    """
    Create and configure a checkpoint callback for model saving.

    Creates the checkpoint directory (removing existing one if present) and returns
    a configured callback for saving models during training. When registry_uploader_callback
    is provided, returns an extended version with support for uploading checkpoints to a remote registry.

    Args:
        dirpath: Path to the directory for saving checkpoints.
        ckpt_cfg: Checkpoint configuration (filename, monitor, mode, save_top_k).
        save_weights_only: If True, only model weights are saved without optimizer and lr-scheduler state.
            Defaults to True.
        registry_uploader_callback: Optional callback for uploading checkpoints to a remote registry.
            Must be specified together with uploading_mode.
        uploading_mode: Checkpoint upload mode:
            - "only-best": only the best checkpoint is uploaded
            - "every-checkpoint": every saved checkpoint is uploaded
            Must be specified together with registry_uploader_callback.

    Returns:
        ModelCheckpointWithRegistryUploader if registry_uploader_callback is provided,
        otherwise standard ModelCheckpoint.

    Raises:
        ValueError: If only one of registry_uploader_callback or uploading_mode is None.

    Note:
        If the dirpath directory already exists, it will be removed and recreated
        (only on the main process in distributed training).

    """
    if (registry_uploader_callback is None) != (uploading_mode is None):
        raise ValueError(
            "Both registry_uploader_callback and uploading_mode must be provided or neither."
        )

    if dirpath.exists():
        if is_main_process():
            logger.warning(f"Checkpoint directory {dirpath} already exists.")
            rmtree(dirpath)
            logger.warning(f"Removed existing checkpoint directory {dirpath}.")
    else:
        logger.info(f"Creating checkpoint directory {dirpath}.")
        dirpath.mkdir(parents=True, exist_ok=True)

    if (registry_uploader_callback is not None) and (uploading_mode is not None):
        checkpoint_callback = ModelCheckpointWithRegistryUploader(
            dirpath=dirpath,
            filename=ckpt_cfg.filename,
            save_top_k=ckpt_cfg.save_top_k,
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
            verbose=True,
            save_weights_only=save_weights_only,
            registry_uploader_callback=registry_uploader_callback,
            uploading_mode=uploading_mode,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            filename=ckpt_cfg.filename,
            save_top_k=ckpt_cfg.save_top_k,
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
            verbose=True,
            save_weights_only=save_weights_only,
        )
    return checkpoint_callback
