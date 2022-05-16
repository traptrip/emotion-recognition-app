import pathlib
from typing import Any, Callable, Optional, Tuple

import cv2
from torchvision.datasets import VisionDataset
from omegaconf import DictConfig


class EmoFace(VisionDataset):
    """FER2013 + CK+48
    <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge>`
    <https://www.kaggle.com/datasets/shawon10/ckplus>

    Args:
        root (string): Root directory of dataset where directory
            ``root/fer2013`` exists.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        label2id: DictConfig = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        base_folder = pathlib.Path(self.root)
        data_folder = base_folder / split
        self._samples = [
            [
                cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB),
                label2id[img_p.parts[-2]] if label2id else img_p.parts[-2],
            ]
            for img_p in data_folder.rglob("*.png")
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image, target = self._samples[idx]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def extra_repr(self) -> str:
        return f"split={self._split}"
