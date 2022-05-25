from typing import List

import torch
import cv2
import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from omegaconf import DictConfig

from application.src.utils import mono_to_stereo


class ImageClassifier:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._img_size = cfg.image_size
        self._transform = A.Compose(
            [A.Resize(*cfg.image_size), A.Normalize(), ToTensorV2()]
        )
        self._model = torch.jit.load(cfg.weights, map_location=torch.device(cfg.device))
        self._model.eval()

    def _prepare(self, faces: List[np.ndarray]) -> torch.Tensor:
        faces = [
            self._transform(
                image=mono_to_stereo(
                    cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)[..., None], -1
                )
            )["image"]
            for face in faces
        ]
        return torch.stack(faces).to(self.cfg.device)

    def predict(self, faces: List[np.ndarray]) -> torch.Tensor:
        with torch.no_grad():
            faces = self._prepare(faces)
            prediction = self._model(faces)
            probas = F.softmax(prediction, dim=1)
        return probas
