import os
import importlib
import random
from copy import deepcopy
from typing import Tuple, Any

import torch
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
from omegaconf import DictConfig, ListConfig


# Config Utils
def set_seed(seed: int = 1234, precision: int = 10) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(precision=precision)


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def load_augs(cfg: DictConfig) -> A.Compose:
    """
    Load albumentations
    Args:
        cfg:
    Returns:
        compose object
    """
    augs = []
    for a in cfg:
        if a["class_name"] == "albumentations.OneOf":
            small_augs = []
            for small_aug in a["params"]:
                # yaml can't contain tuples, so we need to convert manually
                params = {
                    k: (v if not isinstance(v, ListConfig) else tuple(v))
                    for k, v in small_aug["params"].items()
                }
                aug = load_obj(small_aug["class_name"])(**params)
                small_augs.append(aug)
            aug = load_obj(a["class_name"])(small_augs)
            augs.append(aug)

        else:
            params = {
                k: (v if not isinstance(v, ListConfig) else tuple(v))
                for k, v in a["params"].items()
            }
            aug = load_obj(a["class_name"])(**params)
            augs.append(aug)

    return A.Compose(augs)


# Data
def get_dataloaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Function to get dataloader for data specified in config
    """
    train_dataset = load_obj(cfg.datamodule._target_)(cfg, "train")
    valid_dataset = load_obj(cfg.datamodule._target_)(cfg, "val")
    return (
        DataLoader(
            train_dataset,
            batch_size=cfg.general.batch_size,
            shuffle=True,
            num_workers=cfg.general.n_workers,
            pin_memory=True,
        ),
        DataLoader(
            valid_dataset,
            batch_size=cfg.general.batch_size,
            shuffle=False,
            num_workers=cfg.general.n_workers,
            pin_memory=True,
        ),
    )


# Logging
def initialize_logger(logger_cfg: DictConfig) -> Any:
    """Func to initialize logger object"""
    return load_obj(logger_cfg._target_)(**logger_cfg.params)


def log_meta(logger: Any, train_meta: dict) -> None:
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


# Trainer
def initialize_trainer(cfg: DictConfig, logger: Any):
    """Func to initialize trainer object"""
    return load_obj(cfg.trainer._target_)(cfg, logger)


def mono_to_color(X: np.ndarray) -> np.ndarray:
    return np.stack([X, X, X], axis=-1)
