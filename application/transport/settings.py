import os
import yaml
from pathlib import Path
from omegaconf import DictConfig

PROJECT_DIR = Path(__file__).parent.parent
MEDIA_ROOT = PROJECT_DIR / "media"
CONFIG_PATH = os.environ.get("ML_CONFIG_PATH", PROJECT_DIR / ".conf/app_config.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = DictConfig(yaml.safe_load(f))
