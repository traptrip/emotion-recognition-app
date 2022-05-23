import os
import hydra
from omegaconf import DictConfig
from training.train_pipeline import train_pipeline

# CFG_NAME = "train_audio_cfg"
CFG_NAME = os.environ.get("CFG_NAME", default="train_cv_cfg")


@hydra.main(config_path=".conf", config_name=CFG_NAME)
def run_train(config: DictConfig):
    train_pipeline(config)


if __name__ == "__main__":
    run_train()
