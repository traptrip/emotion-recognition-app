import pathlib
from typing import Any, Tuple

import torch
import librosa
import torchaudio
import numpy as np
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset
from omegaconf import DictConfig

from training.utils import stereo_to_mono, mono_to_stereo


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

        self.stage = stage
        self.sample_rate = cfg.general.sample_rate
        base_folder = pathlib.Path(cfg.datamodule.root)
        data_folder = base_folder / stage
        label2id = cfg.datamodule.label2id
        mel_params = cfg.datamodule.mel_params
        amp2db_params = cfg.datamodule.amp2db_params
        self.audio_len = cfg.general.audio_len
        self._mel_creator = MelCreator(mel_params, amp2db_params)
        self._noise_params = cfg.datamodule.noise.params

        if stage == "train":
            noise_folder = base_folder / cfg.datamodule.noise.dir
            self._noises_paths = list(noise_folder.rglob("*.wav"))
            self._noises = [
                stereo_to_mono(torchaudio.load(noise_path)[0])
                for noise_path in noise_folder.rglob("*.wav")
            ]

        self._samples = [
            [
                audio_path,
                label2id[audio_path.parts[-2]] if label2id else audio_path.parts[-2],
            ]
            for audio_path in data_folder.rglob("*.wav")
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        # read audio
        audio_path, target = self._samples[idx]
        # audio, _ = torchaudio.load(audio_path)
        audio, _ = torchaudio.sox_effects.apply_effects_file(
            audio_path,
            effects=[
                # ["pitch", f"{np.random.choice([-1, 0, 1])}"],
                ["rate", f"{self.sample_rate}"],
            ],
        )
        # audio, sr = librosa.load(audio_path)
        # audio = torch.from_numpy(
        #     librosa.effects.pitch_shift(audio, sr, np.random.choice([-1, 0, 1]))
        # )

        # (n,wav_len) -> (1, wav_len) -> (3, wav_len)
        audio = stereo_to_mono(audio, axis=0)
        # audio = mono_to_stereo(audio, axis=0)
        audio = self._pad_audio(audio)

        # transform
        if self.stage == "train":
            noise = self._get_random_noize()
            audio = self._add_noise(audio, noise, **self._noise_params)

        # normalize
        audio = audio / audio.abs().max()
        melspec = self._mel_creator(audio)
        melspec = mono_to_stereo(melspec, axis=0)
        if melspec.isnan().sum() > 0:
            print(audio_path)

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
        idx = np.random.randint(0, len(self._noises))
        return self._noises[idx]

    def _add_noise(self, clean, noise, min_amp, max_amp):
        noise_amp = np.random.uniform(min_amp, max_amp)
        start = np.random.randint(0, noise.shape[1] - clean.shape[1] + 1)
        noise_part = noise[:, start : start + clean.shape[1]]

        if noise_part.abs().max() == 0:
            return clean
        noise_mult = clean.abs().max() / noise_part.abs().max() * noise_amp
        return (clean + noise_part * noise_mult) / (1 + noise_amp)
