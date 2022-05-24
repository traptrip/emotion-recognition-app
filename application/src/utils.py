import logging
import time
from contextlib import contextmanager
from typing import Union


import torch
import numpy as np
import torchaudio.transforms as T


class MelCreator:
    def __init__(self, mel_params, amp2db_params) -> None:
        self.make_melspec = T.MelSpectrogram(**mel_params)
        self.atdb = T.AmplitudeToDB(**amp2db_params)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        melspec = self.atdb(self.make_melspec(audio))
        return melspec


def mono_to_stereo(
    x: Union[np.ndarray, torch.Tensor], axis=0
) -> Union[np.ndarray, torch.Tensor]:
    if x.shape[axis] == 1:
        new_shape = list(x.shape)
        new_shape[axis] = 3
        if isinstance(x, np.ndarray):
            return np.stack([x, x, x], axis=axis).reshape(new_shape)
        elif isinstance(x, torch.Tensor):
            return torch.stack([x, x, x], dim=axis).reshape(new_shape)
    return x


def stereo_to_mono(
    x: Union[np.ndarray, torch.Tensor], axis=0
) -> Union[np.ndarray, torch.Tensor]:
    if x.shape[axis] > 1:
        if isinstance(x, np.ndarray):
            return np.mean(x, keepdims=True)
        elif isinstance(x, torch.Tensor):
            return torch.mean(x, dim=axis, keepdim=True)
    return x


@contextmanager
def time_manager(stage: str):
    start_time = time.time()
    yield
    end_time = time.time()
    recognition_time = end_time - start_time
    logging.info(f"stage: {stage} | time: {recognition_time:.6f}")
