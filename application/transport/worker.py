import os

from celery import Celery
from celery.states import STARTED
from celery.contrib.abortable import AbortableTask

from src.recognizer import EmotionRecognizer
from .settings import CONFIG


celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery.conf.result_backend = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://localhost:6379"
)
emotion_recognizer = EmotionRecognizer(CONFIG)


@celery.task(name="create_task", bind=True, base=AbortableTask)
def create_task(self, video_path: str):
    self.update_state(state=STARTED)
    video_path, df_path = emotion_recognizer.recognize(video_path)
    return [video_path, df_path]
