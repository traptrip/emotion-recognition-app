from typing import Any
from copy import deepcopy

from omegaconf import DictConfig

from utils import load_obj


def initialize_logger(logger_cfg: DictConfig):
    logger = load_obj(logger_cfg._target_)(**logger_cfg.params)
    return logger


def log_meta(logger: Any, train_meta: dict):
    """
    train_meta = {
        "n_iter": int,
        "train": {"loss": float, "metric": float},
        "val": {"loss": float, "metric": float}
    }
    """
    tm = deepcopy(train_meta)
    n_iter = tm["n_iter"]
    del tm["n_iter"]
    for stage, val in tm.items():
        for m, v in val.items():
            logger.add_scalar(f"{m}/{stage}", v, n_iter)
