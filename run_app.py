import hydra
from omegaconf import DictConfig
from application.src.recognizer import EmotionRecognizer


@hydra.main(config_path=".conf", config_name="app_config")
def run(cfg: DictConfig):
    recognizer = EmotionRecognizer(cfg)
    recognizer.recognize(
        "/home/and/projects/OWN/emotion-recognition-app/data/video.mp4"
    )


if __name__ == "__main__":
    run()
