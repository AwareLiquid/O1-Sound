"""Liquid-core keyword spotter.

The recurrent core is a multi-timescale liquid layer: each channel carries its
own learnable time constant tau, parameterised as `softplus(log_tau) + tau_min`
so it stays positive without a clamp. The state update is the discretised form
of

    tau * dh/dt = -h + f(Wx + Uh + b)

which, with a step of 1 frame, gives a per-channel leak `alpha = 1/tau` --
short-tau channels track the current phoneme, long-tau channels hold the shape
of the whole word. That spread is what lets a small model separate a wake phrase
from a phonetically similar one without stacking depth.

Two properties matter for the target (always-on, battery powered):

* Streaming is O(1). The state is a fixed (B, hidden) tensor regardless of how
  long the microphone has been open; `step()` advances one frame at a time and
  `forward()` is the batched-training equivalent of the same recurrence.
* It exports to ONNX. The loop is a plain Python `for` over frames, which the
  TorchScript tracer unrolls at a fixed frame count; `step()` is the graph used
  for real streaming inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class O1SoundConfig:
    n_mels: int = 40
    hidden: int = 192
    n_layers: int = 2
    n_classes: int = 2          # overwritten by the training script
    tau_min: float = 1.0
    tau_max: float = 24.0       # frames; at 10 ms hop this is 10-240 ms
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden <= 0 or self.n_layers <= 0:
            raise ValueError("hidden and n_layers must be positive")
        if not 0.0 < self.tau_min < self.tau_max:
            raise ValueError("require 0 < tau_min < tau_max")


class LiquidCell(nn.Module):
    """One multi-timescale liquid recurrent layer."""

    def __init__(self, d_in: int, hidden: int, tau_min: float, tau_max: float) -> None:
        super().__init__()
        self.hidden = hidden
        self.tau_min = tau_min

        self.inp = nn.Linear(d_in, hidden)
        self.rec = nn.Linear(hidden, hidden, bias=False)

        # Spread initial taus geometrically across [tau_min, tau_max] so the
        # layer starts with a genuine range of timescales rather than having to
        # discover one. Stored in softplus-inverse space.
        taus = torch.logspace(
            torch.log10(torch.tensor(tau_min)),
            torch.log10(torch.tensor(tau_max)),
            hidden,
        )
        self.log_tau = nn.Parameter(torch.log(torch.expm1((taus - tau_min).clamp_min(1e-6))))

    def alpha(self) -> torch.Tensor:
        """Per-channel leak in (0, 1]."""
        tau = F.softplus(self.log_tau) + self.tau_min
        return 1.0 / tau

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Advance one frame. x_t: (B, d_in), h: (B, hidden) -> (B, hidden)."""
        pre = self.inp(x_t) + self.rec(h)
        return h + self.alpha() * (-h + torch.tanh(pre))

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, d_in) -> (B, T, hidden)."""
        b, t, _ = x.shape
        h = torch.zeros(b, self.hidden, device=x.device, dtype=x.dtype) if h0 is None else h0
        out = []
        for i in range(t):
            h = self.step(x[:, i], h)
            out.append(h)
        return torch.stack(out, dim=1)


class O1Sound(nn.Module):
    """Log-mel frames in, keyword logits out."""

    def __init__(self, config: O1SoundConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.n_mels)

        dims = [config.n_mels] + [config.hidden] * config.n_layers
        self.cells = nn.ModuleList(
            LiquidCell(dims[i], dims[i + 1], config.tau_min, config.tau_max)
            for i in range(config.n_layers)
        )
        self.drop = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden, config.n_classes)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, T, n_mels) -> logits (B, n_classes).

        Pools over time by mean, which keeps the decision insensitive to where
        in the window the keyword lands.
        """
        h = self.norm(mel)
        for cell in self.cells:
            h = cell(h)
        return self.head(self.drop(h.mean(dim=1)))

    # ---- streaming -------------------------------------------------------

    def init_state(self, batch: int = 1, device=None, dtype=torch.float32) -> list[torch.Tensor]:
        return [
            torch.zeros(batch, c.hidden, device=device, dtype=dtype) for c in self.cells
        ]

    def step(
        self, mel_t: torch.Tensor, state: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """One frame of streaming inference.

        mel_t: (B, n_mels). Returns (logits, new_state). The carried state is
        `sum(hidden)` floats per stream and never grows with stream length.
        """
        h = self.norm(mel_t)
        new_state = []
        for cell, s in zip(self.cells, state):
            h = cell.step(h, s)
            new_state.append(h)
        return self.head(h), new_state

    def state_bytes(self, batch: int = 1, dtype_size: int = 4) -> int:
        return sum(c.hidden for c in self.cells) * batch * dtype_size

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
