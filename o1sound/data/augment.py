"""Waveform and spectrogram augmentation for wake-word training.

Wake-word data is small and lopsided: MSWC yields a few hundred clips of one
greeting against thousands of other words. Class weighting cannot fix that --
it only tells the loss to care more about the same 219 recordings. Augmentation
is what actually enlarges the positive set, which is why it is standard in
keyword spotting and why its absence showed up as a plateau in run 1.

Every transform here is label-preserving by construction: a shifted, noisier,
slightly faster "hello" is still "hello". Train split only -- applying any of
this to dev or test would measure the augmentation, not the model.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def time_shift(x: np.ndarray, max_frac: float, rng: random.Random) -> np.ndarray:
    """Roll the waveform, zero-filling the vacated end.

    A keyword never lands at the same offset twice in the wild. Trained only on
    centre-aligned clips, a model learns the alignment as much as the word.
    """
    if max_frac <= 0 or len(x) == 0:
        return x
    n = int(len(x) * max_frac)
    if n == 0:
        return x
    k = rng.randint(-n, n)
    if k == 0:
        return x
    out = np.zeros_like(x)
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


def add_noise(x: np.ndarray, snr_db_range: tuple[float, float],
              rng: random.Random) -> np.ndarray:
    """Mix white noise at a random SNR.

    An always-on microphone never hears clean audio. A model shown only clean
    clips calibrates its threshold against a signal level it will not meet.
    """
    sig = float(np.mean(x ** 2))
    if sig <= 0:
        return x
    snr = rng.uniform(*snr_db_range)
    noise_power = sig / (10 ** (snr / 10))
    noise = np.random.randn(len(x)).astype(np.float32) * np.sqrt(noise_power)
    return (x + noise).astype(np.float32)


def speed_perturb(x: np.ndarray, rate_range: tuple[float, float],
                  rng: random.Random) -> np.ndarray:
    """Resample to shift speed and pitch together, then refit the length.

    Linear interpolation on purpose: this is regularisation, not signal
    processing, and speakers vary in rate far more than the interpolation error
    costs.
    """
    rate = rng.uniform(*rate_range)
    if abs(rate - 1.0) < 1e-3 or len(x) == 0:
        return x
    n_out = max(1, int(round(len(x) / rate)))
    idx = np.linspace(0, len(x) - 1, n_out).astype(np.float32)
    y = np.interp(idx, np.arange(len(x), dtype=np.float32), x).astype(np.float32)
    if len(y) >= len(x):
        start = rng.randint(0, len(y) - len(x))
        return y[start:start + len(x)]
    out = np.zeros_like(x)
    start = rng.randint(0, len(x) - len(y))
    out[start:start + len(y)] = y
    return out


def gain(x: np.ndarray, db_range: tuple[float, float], rng: random.Random) -> np.ndarray:
    return (x * (10 ** (rng.uniform(*db_range) / 20))).astype(np.float32)


class WaveformAugment:
    """The waveform chain. Construct once per dataset; train split only."""

    def __init__(self, shift_frac: float = 0.25,
                 snr_db: tuple[float, float] = (5.0, 25.0),
                 speed: tuple[float, float] = (0.9, 1.1),
                 gain_db: tuple[float, float] = (-6.0, 6.0),
                 p: float = 0.8, seed: int = 0) -> None:
        self.shift_frac = shift_frac
        self.snr_db = snr_db
        self.speed = speed
        self.gain_db = gain_db
        self.p = p
        self.rng = random.Random(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.p:
            return x
        x = speed_perturb(x, self.speed, self.rng)
        x = time_shift(x, self.shift_frac, self.rng)
        x = add_noise(x, self.snr_db, self.rng)
        x = gain(x, self.gain_db, self.rng)
        return np.clip(x, -1.0, 1.0).astype(np.float32)


def spec_augment(mel: torch.Tensor, n_freq_masks: int = 1, freq_width: int = 6,
                 n_time_masks: int = 1, time_width: int = 12) -> torch.Tensor:
    """SpecAugment over a (B, T, n_mels) batch. Returns a masked copy.

    Masking forces the model to decide on partial evidence instead of keying on
    one band or one instant -- which is what a duty-cycled microphone hands it
    anyway.
    """
    out = mel.clone()
    b, t, f = out.shape
    for i in range(b):
        for _ in range(n_freq_masks):
            w = int(torch.randint(0, freq_width + 1, (1,)))
            if w and f > w:
                f0 = int(torch.randint(0, f - w, (1,)))
                out[i, :, f0:f0 + w] = 0.0
        for _ in range(n_time_masks):
            w = int(torch.randint(0, time_width + 1, (1,)))
            if w and t > w:
                t0 = int(torch.randint(0, t - w, (1,)))
                out[i, t0:t0 + w, :] = 0.0
    return out
