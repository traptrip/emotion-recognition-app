import os
import logging
import hydra
from omegaconf import DictConfig

from training.utils import (
    set_seed,
    get_dataloaders,
    initialize_logger,
    initialize_trainer,
)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s:%(message)s", datefmt="%d:%m:%Y|%H:%M:%S")


def train_pipeline(cfg: DictConfig) -> None:
    os.makedirs(cfg.general.checkpoint_path, exist_ok=True)
    set_seed(cfg.general.seed)
    logger = initialize_logger(cfg.logger)
    train_loader, val_loader = get_dataloaders(cfg)
    trainer = initialize_trainer(cfg, logger)
    trainer.train(train_loader, val_loader)


@hydra.main(config_path=".conf", config_name="train_image_classification")
def run(cfg: DictConfig):
    train_pipeline(cfg)


if __name__ == "__main__":
    run()
