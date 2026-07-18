from lightning.pytorch.callbacks import EarlyStopping

from kostyl.ml.configs import EarlyStoppingConfig


def setup_early_stopping_callback(
    early_stopping_cfg: EarlyStoppingConfig,
) -> EarlyStopping:
    """Sets up an EarlyStopping callback for PyTorch Lightning based on the provided configuration."""
    early_stopping_callback = EarlyStopping(
        monitor=early_stopping_cfg.monitor,
        mode=early_stopping_cfg.mode,
        patience=early_stopping_cfg.patience,
        min_delta=early_stopping_cfg.min_delta,
        strict=True,
        verbose=True,
    )
    return early_stopping_callback
