import logging
import hydra
from omegaconf import DictConfig

from training.emotion_classification.image.model import initialize_model
from training.logger import initialize_logger
from training.emotion_classification.image.data.dataset import get_dataloaders
from training.emotion_classification.image.model import train_model
from utils import set_seed

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s:%(message)s", datefmt="%d:%m:%Y|%H:%M:%S")


def train_pipeline(cfg: DictConfig) -> None:
    set_seed(cfg.general.seed)
    logger = initialize_logger(cfg.logger)
    model = initialize_model(cfg.model)
    train_loader, val_loader = get_dataloaders(cfg)
    train_model(cfg, logger, model, train_loader, val_loader)


@hydra.main(config_path=".conf", config_name="train_image_classification")
def run(cfg: DictConfig):
    train_pipeline(cfg)


if __name__ == "__main__":
    run()
