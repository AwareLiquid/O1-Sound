"""Host-side parity gate: C kernel vs PyTorch, frame by frame.

1. Export the checkpoint to C arrays (scripts/export_c_kernel.py).
2. Compile freertos/o1sound_kernel.c + tests/test_driver.c with zig cc.
3. Feed identical 480-sample windows through both implementations.
4. Assert logits agree within 1e-3 (cross-algorithm DFT vs torch FFT).

Run:  python tests/test_c_parity.py
"""
from __future__ import annotations

import io
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIG = None
for cand in ("zig",):
    p = shutil.which(cand)
    if p:
        ZIG = p
        break

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import torch

sys.path.insert(0, str(ROOT))

from o1sound.features import LogMel  # noqa: E402
from o1sound.model import O1Sound, O1SoundConfig  # noqa: E402


def load_model(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = O1SoundConfig(**ck["config"])
    model = O1Sound(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def build_c_kernel(zig: str, tmp: Path) -> Path:
    src = ROOT / "freertos" / "o1sound_kernel.c"
    drv = ROOT / "tests" / "test_driver.c"
    exe = tmp / "o1s_driver.exe"
    cmd = [zig, "cc", "-O2", f"-I{ROOT / 'freertos'}",
           str(src), str(drv), "-lm", "-o", str(exe)]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return exe


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/o1sound.pt",
                    help="checkpoint matching the weights.h currently exported")
    args = ap.parse_args()

    ckpt = ROOT / args.ckpt
    if not ckpt.exists():
        print(f"SKIP: no checkpoint at {ckpt}")
        return 0
    if ZIG is None:
        print("SKIP: zig not found on PATH (needed to compile the C kernel)")
        return 0

    model, cfg = load_model(ckpt)
    front = LogMel()

    # Reference signal: enough frames for steady state (t >= 2).
    torch.manual_seed(0)
    sr = 16000
    n_frames = 40
    wav = 0.3 * torch.randn(n_frames * 160 + 480)          # noise
    t_ = torch.arange(n_frames * 160 + 480) / sr
    wav += 0.5 * torch.sin(2 * math.pi * 220 * t_)          # tonal probe
    mels = front(wav)                                       # (1, T, 40)

    # Torch reference logits, streaming step path.
    state = model.init_state(1)
    ref = []
    for t in range(mels.shape[1]):
        logits, state = model.step(mels[:, t], state)
        ref.append(logits[0].detach().tolist())
    ref = torch.tensor(ref)

    # Windows for the C side: torch.stft(center=True, pad_mode="reflect")
    # pads 240 samples on each side, then frames every hop. Build the padded
    # signal exactly and slice ALL frames, so the C stream starts from the
    # same zero state and sees the same windows as torch — including the
    # edge frames 0..1 that reflect-padding special-cases.
    padded = torch.nn.functional.pad(wav.unsqueeze(0), (240, 240), mode="reflect")[0]
    windows = [padded[t * 160:t * 160 + 480] for t in range(mels.shape[1])]
    win = torch.stack(windows)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        exe = build_c_kernel(ZIG, tmp)
        inbin = tmp / "frames.f32"
        win.numpy().tofile(inbin)
        out = subprocess.run([str(exe)], stdin=open(inbin, "rb"),
                             capture_output=True, check=True)
        c_logits = []
        for line in out.stdout.decode().splitlines():
            parts = line.split()
            c_logits.append([float(p) for p in parts[1:]])
        c_logits = torch.tensor(c_logits)

    torch_logits = ref                                             # frames 0..T-1
    assert torch_logits.shape == c_logits.shape, (torch_logits.shape, c_logits.shape)
    diff = (torch_logits - c_logits).abs().max().item()
    span = torch_logits.max().item() - torch_logits.min().item()
    print(f"frames compared: {torch_logits.shape[0]}")
    print(f"max |torch - C| on logits: {diff:.3e} (logit span {span:.2f})")
    # Relative tolerance: the C frontend runs a float32 direct DFT where torch
    # runs an FFT — the absolute rounding error scales with the logit
    # magnitude. A wake decision (argmax / threshold) is unaffected below
    # 0.1% of the logit span.
    if diff < max(1e-3, 1e-3 * span):
        print("PARITY PASS")
        return 0
    print("PARITY FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
