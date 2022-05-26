import os

from celery import Celery

from src.recognizer import EmotionRecognizer
from .settings import CONFIG


celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery.conf.result_backend = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://localhost:6379"
)
emotion_recognizer = EmotionRecognizer(CONFIG)


@celery.task(name="create_task")
def create_task(video_path: str):
    video_path, df_path = emotion_recognizer.recognize(video_path)
    return [video_path, df_path]
