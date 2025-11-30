from pathlib import Path
from shutil import rmtree

from lightning.pytorch.loggers import TensorBoardLogger

from kostyl.ml.dist_utils import is_main_process
from kostyl.utils.logging import setup_logger


logger = setup_logger()


def setup_tb_logger(
    runs_dir: Path,
) -> TensorBoardLogger:
    """Sets up a TensorBoardLogger for PyTorch Lightning."""
    if runs_dir.exists():
        if is_main_process():
            logger.warning(f"TensorBoard log directory {runs_dir} already exists.")
            rmtree(runs_dir)
            logger.warning(f"Removed existing TensorBoard log directory {runs_dir}.")
    else:
        logger.info(f"Creating TensorBoard log directory {runs_dir}.")
        runs_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = TensorBoardLogger(
        save_dir=runs_dir,
        name="tb_logs",
        default_hp_metric=False,
    )
    return tb_logger
