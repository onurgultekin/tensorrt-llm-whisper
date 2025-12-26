# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal subset from TensorRT-LLM Whisper example:
# https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/whisper
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Optional, Union

import numpy as np
import soundfile
import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160


def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio_wav_format(wav_path: str):
    assert wav_path.endswith(".wav"), f"Only support .wav format, got: {wav_path}"
    waveform, sample_rate = soundfile.read(wav_path)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz")
    return waveform, sample_rate


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int, mel_filters_dir: str | None = None) -> torch.Tensor:
    if n_mels not in {80, 128}:
        raise ValueError(f"Unsupported n_mels: {n_mels}")
    if mel_filters_dir is None:
        mel_filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    else:
        mel_filters_path = os.path.join(mel_filters_dir, "mel_filters.npz")
    with np.load(mel_filters_path) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    mel_filters_dir: str | None = None,
):
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            if audio.endswith(".wav"):
                audio, _ = load_audio_wav_format(audio)
            else:
                audio = load_audio(audio)
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Unsupported audio type: {type(audio)}")
        audio = torch.from_numpy(audio.astype(np.float32))

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels, mel_filters_dir)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

