import os
import random
import importlib
from typing import Any

import torch
import albumentations as A
import numpy as np
from omegaconf import DictConfig, ListConfig


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


def mono_to_color(X: np.ndarray):
    return np.stack([X, X, X], axis=-1)
