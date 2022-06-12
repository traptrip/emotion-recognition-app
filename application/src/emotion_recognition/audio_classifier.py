import math
import torch
import numpy as np
import torch.nn.functional as F
import torchaudio.transforms as T
from omegaconf import DictConfig
from moviepy.editor import AudioClip

from ..utils import mono_to_stereo, stereo_to_mono, MelCreator


class AudioClassifier:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._mel_creator = MelCreator(cfg.mel_params, cfg.amp2db_params)
        self._audio_size = cfg.audio_size
        self._clip_size = int(cfg.audio_size / cfg.mel_params.sample_rate)
        self._model = torch.jit.load(cfg.weights, map_location=torch.device(cfg.device))
        self._model.eval()
        self.chunk_size = 128
        self.n_classes = 7

    def _prepare(self, audio_clip: AudioClip) -> T.MelSpectrogram:
        audio = torch.from_numpy(audio_clip.to_soundarray().T)
        audio = stereo_to_mono(audio)
        audio = self._pad_audio(audio)
        audio = audio / audio.abs().max()
        melspec = self._mel_creator(audio.float())
        melspec = mono_to_stereo(melspec).unsqueeze(0)
        melspec = melspec.to(self.cfg.device)
        return melspec

    def _pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if self._audio_size - audio.shape[-1] > 0:
            i = np.random.randint(0, self._audio_size - audio.shape[-1])
        else:
            i = 0
        pad_patern = (i, self._audio_size - audio.shape[-1] - i)
        audio = F.pad(audio, pad_patern, "constant").detach()
        return audio

    def predict(self, audio: AudioClip) -> torch.Tensor:
        with torch.no_grad():
            melspec = self._prepare(audio)
            prediction = self._model(melspec)
            probas = F.softmax(prediction, dim=1)
        return probas

    def predict_chunks(self, audio: AudioClip) -> torch.Tensor:
        full_clip_size = audio.duration
        input_data = torch.stack(
            [
                self._prepare(
                    audio.subclip(i, min(i + self._clip_size, int(full_clip_size)))
                )[0]
                for i in range(0, int(full_clip_size), self._clip_size)
            ]
            + [self._prepare(audio.subclip(-(full_clip_size % self._clip_size)))[0]]
            if full_clip_size % self._clip_size
            else []
        )
        input_data = input_data.to(self.cfg.device)

        data_len = len(input_data)
        n_chunks = math.ceil(data_len / self.chunk_size)
        all_probas = torch.zeros((data_len, self.n_classes))
        with torch.no_grad():
            for i in range(n_chunks):
                batch = input_data[i * self.chunk_size : (i + 1) * self.chunk_size]
                prediction = self._model(batch)
                probas = F.softmax(prediction, dim=1)
                all_probas[i * self.chunk_size : (i + 1) * self.chunk_size] = probas
        return all_probas
