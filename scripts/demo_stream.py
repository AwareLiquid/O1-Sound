"""demo_stream.py — O1-Sound 实时唤醒词检测演示

mic → 16kHz 流式 log-mel → 液态核心 step()（O(1) 状态）→ wake 概率 → 触发

用法:
  python scripts/demo_stream.py                 # 默认麦克风, 阈值 0.9
  python scripts/demo_stream.py --threshold 0.95
  python scripts/demo_stream.py --wav path.wav  # 无麦克风: 回放 wav 走同一管线
  python scripts/demo_stream.py --list-devices  # 列出音频设备

说明:
  - 默认 checkpoint: checkpoints/o1sound.pt（多语言 Run 7, dev acc 0.885）
  - wake 概率 = softmax(logits)[1]（类别 0=背景, 1=唤醒词）
  - 状态恒定: 2×640 float = 5,120 字节, 与流长无关
  - 触发后有 1.5s 冷却, 避免同一句话重复触发

诚实边界: 这是研究级模型——英语实测 FRR 0.146 @ FAR 0.046 (Run 2),
多语言最差语言 FRR 1.000。演示用途, 不是商用唤醒引擎。
"""
import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from o1sound.features import LogMel  # noqa: E402
from o1sound.model import O1Sound, O1SoundConfig  # noqa: E402

SAMPLE_RATE = 16000
HOP = 160          # 10 ms
WINDOW = 480       # 30 ms
COOLDOWN_S = 1.5


def load_model(ckpt_path: str) -> O1Sound:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = O1SoundConfig(**ckpt["config"])
    model = O1Sound(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    return model.eval()


class StreamingDetector:
    """流式唤醒检测: 样本缓冲 + 帧级 log-mel + step() + 冷却触发。"""

    def __init__(self, model: O1Sound, frontend: LogMel, threshold: float,
                 cooldown_s: float = COOLDOWN_S):
        self.model = model
        self.frontend = frontend
        self.threshold = threshold
        self.cooldown_frames = int(cooldown_s * SAMPLE_RATE / HOP)
        self.state = model.init_state(1)
        self.buffer = np.zeros(WINDOW, dtype=np.float32)
        self.filled = 0
        self.since_fire = self.cooldown_frames  # 启动即视为刚触发过(免初始噪声误报)
        self.fires = 0

    def _mel_frame(self) -> torch.Tensor:
        """当前 480 样本窗 → 1 帧 (1, 40) log-mel（center=False 流式口径）。"""
        wav = torch.from_numpy(self.buffer).unsqueeze(0)  # (1, 480)
        spec = torch.stft(
            wav, n_fft=self.frontend.n_fft, hop_length=self.frontend.hop_length,
            window=self.frontend.window, center=False, onesided=True,
            return_complex=True,
        )
        power = spec.real.pow(2) + spec.imag.pow(2)      # (1, n_freqs, 1)
        mel = torch.matmul(self.frontend.fb, power)       # (1, 40, 1)
        return torch.log(mel + self.frontend.eps).squeeze(2)  # (1, 40)

    def push(self, samples: np.ndarray) -> None:
        """送入任意长度的 float32 样本（16kHz, 取值 [-1,1]）。"""
        for s in samples:
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = s
            self.filled += 1
            if self.filled < WINDOW:
                continue
            if (self.filled - WINDOW) % HOP != 0:
                continue
            self.since_fire += 1
            with torch.no_grad():
                frame = self._mel_frame()
                logits, self.state = self.model.step(frame, self.state)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
            if prob >= self.threshold and self.since_fire >= self.cooldown_frames:
                self.fires += 1
                self.since_fire = 0
                print(f"  ⏰ WAKE #{self.fires}  (p={prob:.3f})", flush=True)


def run_mic(args, detector):
    import sounddevice as sd
    if args.list_devices:
        print(sd.query_devices())
        return
    print(f"监听中... 说唤醒词 (hello/hallo/hola/bonjour 等), "
          f"Ctrl+C 退出, 阈值={args.threshold}")
    idx = None
    if args.device is not None:
        idx = args.device
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=HOP,
                            device=idx,
                            callback=lambda indata, frames, t, status:
                            detector.push(indata[:, 0])):
            threading.Event().wait()
    except KeyboardInterrupt:
        print("\n退出。")


def run_wav(args, detector):
    import wave
    with wave.open(args.wav, "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, (
            f"需要 16kHz wav, 实际 {w.getframerate()}Hz (可用 ffmpeg -ar 16000 转)")
        n = w.getnframes()
        raw = w.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    print(f"回放 {args.wav} ({len(samples)/SAMPLE_RATE:.1f}s), "
          f"阈值={args.threshold}")
    for i in range(0, len(samples), HOP):
        detector.push(samples[i:i + HOP])
    print(f"完成。共触发 {detector.fires} 次。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/o1sound.pt")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--wav", help="回放 wav 文件（无麦克风测试用）")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    assert Path(args.ckpt).exists(), f"checkpoint 不存在: {args.ckpt}"
    model = load_model(args.ckpt)
    frontend = LogMel()
    detector = StreamingDetector(model, frontend, args.threshold)
    print(f"[O1-Sound] 参数 {model.num_parameters():,} · 携带状态 "
          f"{model.state_bytes(1):,} 字节 · 常量")
    if args.wav:
        run_wav(args, detector)
    else:
        run_mic(args, detector)
    return 0


if __name__ == "__main__":
    sys.exit(main())
