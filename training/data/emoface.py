import pathlib
from typing import Any, Tuple

import cv2
from torch.utils.data import Dataset
from omegaconf import DictConfig

from training.utils import load_augs


class EmoFace(Dataset):
    """FER2013 + CK+48
    <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge>`
    <https://www.kaggle.com/datasets/shawon10/ckplus>

    Args:
        cfg (DictConfig): Config.
        stage (string): The dataset split, supports ``"train"`` (default), or ``"val"``.
    """

    def __init__(self, cfg: DictConfig, stage: str = "train") -> None:
        super().__init__()

        data_folder = pathlib.Path(cfg.datamodule.root) / stage
        label2id = cfg.datamodule.label2id if "label2id" in cfg.datamodule else None
        self._samples = [
            [
                cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB),
                label2id[img_p.parts[-2]] if label2id else img_p.parts[-2],
            ]
            for img_p in data_folder.rglob("*.png")
        ]
        self.transform = (
            load_augs(cfg.augmentations.train.augs)
            if stage == "train"
            else load_augs(cfg.augmentations.valid.augs)
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image, target = self._samples[idx]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, target

    def extra_repr(self) -> str:
        return f"split={self._split}"
