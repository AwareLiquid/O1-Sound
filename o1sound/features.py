"""Log-mel frontend for wake-word spotting.

Implemented with `torch.stft` and a hand-built mel filterbank rather than
torchaudio, for two reasons: it keeps the dependency set to torch alone, and it
traces cleanly to ONNX (torchaudio's transforms historically export poorly).

The frontend is deliberately fixed-function -- no learnable parameters -- so the
7 MB budget is spent entirely on the recurrent core.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def hz_to_mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int, n_fft: int, sample_rate: int, f_min: float, f_max: float
) -> torch.Tensor:
    """Slaney-style triangular filterbank, shape (n_mels, n_fft // 2 + 1)."""
    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(0, sample_rate / 2, n_freqs)

    m_min, m_max = hz_to_mel(f_min), hz_to_mel(f_max)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = torch.tensor([mel_to_hz(float(m)) for m in m_pts])

    # (n_freqs, n_mels + 2) distance from each FFT bin to each mel point
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down = -slopes[:, :-2] / (f_pts[1:-1] - f_pts[:-2]).clamp_min(1e-8)
    up = slopes[:, 2:] / (f_pts[2:] - f_pts[1:-1]).clamp_min(1e-8)
    fb = torch.clamp(torch.minimum(down, up), min=0.0)
    return fb.T.contiguous()


class LogMel(nn.Module):
    """Waveform (B, T) -> log-mel (B, n_frames, n_mels).

    Defaults follow the usual keyword-spotting recipe: 16 kHz, 30 ms window,
    10 ms hop, 40 mel bands over 20-7600 Hz.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 480,
        hop_length: int = 160,
        n_mels: int = 40,
        f_min: float = 20.0,
        f_max: float = 7600.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.eps = eps

        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer(
            "fb", mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max), persistent=False
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spec.real.pow(2) + spec.imag.pow(2)      # (B, n_freqs, n_frames)
        mel = torch.matmul(self.fb, power)                # (B, n_mels, n_frames)
        return torch.log(mel + self.eps).transpose(1, 2)  # (B, n_frames, n_mels)

    def num_frames(self, num_samples: int) -> int:
        return num_samples // self.hop_length + 1
