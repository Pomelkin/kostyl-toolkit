from pathlib import Path
from shutil import rmtree

from lightning.pytorch.callbacks import ModelCheckpoint

from kostyl.ml.configs import CheckpointConfig
from kostyl.ml.dist_utils import is_main_process
from kostyl.utils import setup_logger


logger = setup_logger("callbacks/checkpoint.py")


def setup_checkpoint_callback(
    dirpath: Path,
    ckpt_cfg: CheckpointConfig,
    save_weights_only: bool = True,
) -> ModelCheckpoint:
    """
    Sets up a ModelCheckpoint callback for PyTorch Lightning.

    This function prepares a checkpoint directory and configures a ModelCheckpoint
    callback based on the provided configuration. If the directory already exists,
    it is removed (only by the main process) to ensure a clean start. Otherwise,
    the directory is created.

    Args:
        dirpath (Path): The path to the directory where checkpoints will be saved.
        ckpt_cfg (CheckpointConfig): Configuration object containing checkpoint
            settings such as filename, save_top_k, monitor, and mode.
        save_weights_only (bool, optional): Whether to save only the model weights
            or the full model. Defaults to True.

    Returns:
        ModelCheckpoint: The configured ModelCheckpoint callback instance.

    """
    if dirpath.exists():
        if is_main_process():
            logger.warning(f"Checkpoint directory {dirpath} already exists.")
            rmtree(dirpath)
            logger.warning(f"Removed existing checkpoint directory {dirpath}.")
    else:
        logger.info(f"Creating checkpoint directory {dirpath}.")
        dirpath.mkdir(parents=True, exist_ok=True)

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
