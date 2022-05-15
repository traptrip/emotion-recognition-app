import os

import hydra
from omegaconf import DictConfig

from training.emotion_classification.image.train import (
    train_pipeline as train_image_classification_pipeline,
)


@hydra.main(config_path=".conf", config_name="train_image_classification")
def train_image_classification(config: DictConfig):
    train_image_classification_pipeline(config)


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    train_image_classification()
