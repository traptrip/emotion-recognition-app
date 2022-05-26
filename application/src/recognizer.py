import json
from pathlib import Path
from typing import List

import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from moviepy.editor import VideoClip, VideoFileClip
from moviepy.video.io.bindings import mplfig_to_npimage

from .face_search.retina_torch.retina_api import RetinaDetector
from .emotion_recognition.image_classifier import ImageClassifier
from .emotion_recognition.audio_classifier import AudioClassifier
from .utils import time_manager

torch.multiprocessing.set_start_method("spawn", force=True)
CURRENT_DIR = Path(__file__).parent


class EmotionRecognizer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._clip_size = int(cfg.general.clip_size)
        with open(cfg.recognition.face_detector.model_config_path, "r") as fin:
            retina_cfg = json.load(fin)
        self.face_detector = RetinaDetector(retina_cfg)
        self.emotion_image_classifier = ImageClassifier(
            cfg.recognition.image_classifier
        )
        self.emotion_audio_classifier = AudioClassifier(
            cfg.recognition.audio_classifier
        )
        self._id2label = cfg.recognition.id2label
        self.cur_frame = 0
        self.eps = 0.4

    def recognize(self, video_path: str):
        video_path = Path(video_path)
        save_path_video = str(
            video_path.parent
            / video_path.name.replace(video_path.suffix, f"_result{video_path.suffix}")
        )
        save_path_df = str(
            video_path.parent
            / video_path.name.replace(video_path.suffix, f"_result.csv")
        )
        recognition_result = []
        video_clip = VideoFileClip(
            str(video_path), audio_fps=self.cfg.general.sample_rate
        )
        video_size = video_clip.duration

        # Recognition
        with time_manager("Recognition"):
            for i in range(0, int(video_size), self._clip_size):
                clip = video_clip.subclip(i, min(i + self._clip_size, video_size))
                recognition_result.extend(self._recognize_clip(clip))
            clip_residue_size = video_size % self._clip_size
            if clip_residue_size > 0 and self.eps < clip_residue_size:
                clip = video_clip.subclip(-clip_residue_size)
                recognition_result.extend(self._recognize_clip(clip))

        # Render & Save Clip
        with time_manager("Rendering"):
            self._render_video(video_clip, recognition_result, save_path_video)

        df = self._generate_df(recognition_result)
        df.to_csv(save_path_df, index=False)

        return save_path_video, save_path_df

    def _recognize_clip(self, clip: VideoClip):
        clip_res = []
        for frame in clip.iter_frames():
            faces_coords, faces = self.face_detector.get_faces(frame)
            if len(faces_coords) > 0:
                cv_probas = self.emotion_image_classifier.predict(faces)
            clip_res.append(
                {
                    "cv": self._process_probas(cv_probas)
                    if len(faces_coords) > 0
                    else self._generate_zero_prediction(),
                    "coords": faces_coords,
                    "audio": None,
                }
            )
        audio_probas = self._process_probas(
            self.emotion_audio_classifier.predict(clip.audio)
        )
        for r in clip_res:
            r.update({"audio": audio_probas})
        return clip_res

    def _process_probas(self, probas: torch.Tensor) -> dict:
        return [
            {self._id2label[i]: p.item() for i, p in enumerate(proba)}
            for proba in probas
        ]

    def _render_video(
        self, video_clip: VideoFileClip, recognition_result: List[dict], save_path: str
    ):
        full_probas = [
            {
                emotion: (2 * cv_proba + audio_proba) / 3
                for (emotion, cv_proba), (emotion, audio_proba) in zip(
                    rec["cv"][0].items(), rec["audio"][0].items()
                )
            }
            for rec in recognition_result
        ]
        sub_duration = self.cfg.recognition.sub_duration
        fig, (video, schedule) = plt.subplots(2, 1, figsize=(12, 10))
        self.cur_frame = 0

        def _render_frame(t):
            frame = video_clip.get_frame(t)
            coords = recognition_result[self.cur_frame]["coords"]
            coords = coords[0] if coords else None
            emotions = full_probas[
                max(0, self.cur_frame - sub_duration) : self.cur_frame + 1
            ]
            if coords:
                cv2.rectangle(
                    frame,
                    tuple(coords[:2]),
                    tuple(coords[2:4]),
                    color=(0, 255, 0),
                    thickness=2,
                )
                self._draw_emotions(frame, coords, emotions[-1])

            schedule.clear()
            self._plot_schedule(schedule, emotions)
            video.clear()
            video.imshow(frame)
            video.set_xticks([])
            video.set_yticks([])
            # plt.savefig("result.png")

            self.cur_frame += 1
            return mplfig_to_npimage(fig)

        rendered_video = VideoClip(
            _render_frame, duration=video_clip.duration - self.eps
        )
        rendered_video = rendered_video.set_audio(video_clip.audio)
        rendered_video.write_videofile(
            save_path,
            video_clip.fps,
            audio_fps=self.cfg.general.sample_rate,
            audio_nbytes=2,
            threads=4,
            verbose=False,
            logger=None,
        )
        video_clip.close()
        rendered_video.close()

    @staticmethod
    def _draw_emotions(frame, coords, emotions):
        offset = 0
        max_proba = max(emotions.values())
        for emotion, proba in emotions.items():
            color = (0, 255, 0) if proba == max_proba else (255, 255, 0)
            frame = cv2.putText(
                frame,
                f"{emotion}: {proba:.2f}",
                (coords[2] + 1, coords[1] + 10 + offset),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=color,
                thickness=2,
            )
            offset += 25
        return frame

    def _plot_schedule(self, schedule, probas: List[dict]):
        plots = {emotion: [] for emotion in self._id2label.values()}
        for proba in probas:
            for emotion, p in proba.items():
                plots[emotion].append(p)
        for emotion, values in plots.items():
            color = self.cfg.visualization.label2color[emotion]
            schedule.plot(values, color=color, label=emotion)
        schedule.legend(loc="upper left")

    def _generate_zero_prediction(self):
        return [{emotion: 0 for emotion in self._id2label.values()}]

    def _generate_df(self, recognition_result: List[dict]) -> pd.DataFrame:
        result = []
        for r in recognition_result:
            cv_probas = {f"cv_{emotion}": p for emotion, p in r["cv"]}
            audio_probas = {f"audio_{emotion}": p for emotion, p in r["audio"]}
            result.append({"coords": r["coords"], **cv_probas, **audio_probas})
        return pd.DataFrame(result)
