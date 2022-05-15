from typing import Tuple, Any
from torchvision.datasets import FER2013
from utils import mono_to_color


class FER2013Dataset(FER2013):
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_tensor, target = self._samples[idx]
        image = image_tensor.numpy()

        if self.transform is not None:
            image = mono_to_color(image)
            image = self.transform(image=image)["image"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
