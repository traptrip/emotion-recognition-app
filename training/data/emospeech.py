import pathlib
from typing import Any, Tuple

import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset
from omegaconf import DictConfig

from training.utils import mono_to_color


class MelCreator:
    def __init__(self, mel_params, amp2db_params) -> None:
        self.make_melspec = T.MelSpectrogram(**mel_params)
        self.atdb = T.AmplitudeToDB(**amp2db_params)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        melspec = self.atdb(self.make_melspec(audio))
        return melspec


class EmoSpeech(Dataset):
    """
    1. https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
    2. https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
    3. https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee

    Args:
        cfg (DictConfig): Config.
        stage (string): The dataset split, supports ``"train"`` (default), or ``"valid"``.
    """

    def __init__(self, cfg: DictConfig, stage: str = "train") -> None:
        super().__init__()

        base_folder = pathlib.Path(cfg.datamodule.root)
        data_folder = base_folder / stage
        label2id = cfg.datamodule.label2id
        mel_params = cfg.datamodule.mel_params
        amp2db_params = cfg.datamodule.amp2db_params
        self._ausio_len = cfg.datamodule.audio_len

        if stage == "train":
            noise_folder = base_folder / noise_folder
            self._noises_paths = list(noise_folder.rglob("*.wav"))

        self._samples = [
            [
                audio_path,
                label2id[audio_path.parts[-2]] if label2id else audio_path.parts[-2],
            ]
            for audio_path in data_folder.rglob("*.wav")
        ]

        self._mel_creator = MelCreator(mel_params, amp2db_params)
        self._audio_len = audio_len

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        # read audio
        audio_path, target = self._samples[idx]
        audio, _ = torchaudio.load(audio_path)
        audio = self._pad_audio(audio)

        # transform
        noise = self._get_random_noize()
        audio = self._add_noise(audio, noise, 0, 6)

        # normalize
        audio = audio / audio.abs().max()
        melspec = self.mel_creator(audio)
        melspec = mono_to_color(melspec)

        return melspec, target

    def _pad_audio(self, audio):
        if self.audio_len - audio.shape[-1] > 0:
            i = np.random.randint(0, self.audio_len - audio.shape[-1])
        else:
            i = 0
        pad_patern = (i, self.audio_len - audio.shape[-1] - i)
        audio = F.pad(audio, pad_patern, "constant").detach()
        return audio

    def _get_random_noize(self):
        idx = np.random.randint(0, len(self._noises_paths))
        noise, _ = torchaudio.load(self._noises_paths[idx])
        if noise.shape[0] > 1:
            noise = noise[np.random.randint(0, noise.shape[0])]
            noise = noise.unsqueeze(0)
        return noise

    def _add_noise(self, clean, noise, min_amp, max_amp):
        noise_amp = np.random.uniform(min_amp, max_amp)
        start = np.random.randint(0, noise.shape[1] - clean.shape[1] + 1)
        noise_part = noise[:, start : start + clean.shape[1]]

        if noise_part.abs().max() == 0:
            return clean
        noise_mult = clean.abs().max() / noise_part.abs().max() * noise_amp
        return (clean + noise_part * noise_mult) / (1 + noise_amp)
