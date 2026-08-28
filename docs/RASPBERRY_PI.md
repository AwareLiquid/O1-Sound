# O1-Sound on Raspberry Pi

Run the O1-Sound wake-word spotter on a Raspberry Pi. The model's **constant
carried state (5,120 bytes)** is the whole point: the microphone can stay open
for hours and memory never grows.

> **Honest status.** This is a *research prototype*, not a production wake-word
> engine. English shows a usable signal (FRR 0.146 @ FAR 0.046, Run 2); the
> multilingual claim is not yet supported. It is a great demo of the liquid-core
> O(1) memory property — treat it as a technology demonstration.

## What you need

| Item | Requirement |
|---|---|
| Board | Raspberry Pi 3B+ / 4B / Zero 2 W (ARMv8) — anything that runs 64-bit Raspberry Pi OS |
| Microphone | Any USB microphone (or a ReSpeaker / USB dongle). Onboard 3.5mm mic jack is poor quality |
| OS | Raspberry Pi OS (64-bit), Bookworm or newer |
| Speaker (optional) | To hear the trigger — not required for the demo |

## 1. Install Raspberry Pi OS

Flash 64-bit Raspberry Pi OS to an SD card ([Raspberry Pi Imager](https://www.raspberrypi.com/software/)).
Enable SSH during first boot (Imager settings) so you can work over the network:

```bash
ssh pi@<your-pi-ip>
```

## 2. Install Python deps

Raspberry Pi OS ships Python 3.11. Install a venv so nothing touches the system:

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip libportaudio2 portaudio19-dev
python3 -m venv ~/o1sound-venv
source ~/o1sound-venv/bin/activate
pip install --upgrade pip
```

## 3. Get the code + model

```bash
cd ~
git clone https://github.com/AwareLiquid/O1-Sound.git
cd O1-Sound
pip install -e .
pip install onnxruntime sounddevice soundfile
```

Download the published ONNX checkpoint (4.8 MB) from the research release:

```bash
mkdir -p checkpoints
wget -O checkpoints/o1sound.onnx \
  https://github.com/AwareLiquid/O1-Sound/releases/download/research-2026-08-18/o1sound.onnx
```

> **Why ONNX and not the .pt?** ONNX Runtime has first-class ARM support and
> the exported graph is exactly the streaming `step()` — one log-mel frame in,
> logits + next state out. The `.pt` checkpoint would need PyTorch (heavy on a Pi).

## 4. Quick verification (no microphone)

Generate a test tone or use any 16 kHz WAV. If you don't have one, synthesize a
short "hello" is not trivial on the CLI — instead verify the pipeline with a
recorded clip, or just proceed to the microphone step and test live.

To confirm the model loads and the ONNX graph runs:

```bash
python -c "
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession('checkpoints/o1sound.onnx', providers=['CPUExecutionProvider'])
mel = np.zeros((1, 40), dtype=np.float32)          # one log-mel frame
state = [np.zeros((1, 640), dtype=np.float32) for _ in range(2)]
out = sess.run(None, {'mel_frame': mel, 'state_in_0': state[0], 'state_in_1': state[1]})
print('logits', out[0].shape, 'state_out_0', out[1].shape)
"
```

Expected: `logits (1, 2) state_out_0 (1, 640)` — the constant 640-float state
per layer is what stays flat no matter how long the mic stays open.

## 5. Live microphone demo

```bash
python scripts/demo_stream.py --list-devices    # find your mic index
python scripts/demo_stream.py --device 0        # use your mic, threshold 0.9
```

Say "hello" (or "hallo" / "hola" / "bonjour"). When the model fires you'll see:

```
  ⏰ WAKE #1  (p=0.932)
```

The detector prints the model's carried-state size on startup:

```
[O1-Sound] 参数 1,298,064 · 携带状态 5,120 字节 · 常量
```

**5,120 bytes constant** — that is the property this architecture exists for.

## 6. What is actually running

```
mic (16 kHz) → log-mel (40 dims, torch.stft) → liquid core step() → wake / not-wake
                                                ↑ state: 5,120 B, constant
```

- **Streaming is O(1).** The state is a fixed `(batch, hidden)` tensor per layer.
  It does not grow with how long the microphone has been open.
- **`step()` is what ships.** The ONNX graph is the single-frame step, not a
  fixed-length window — an always-on device runs this forever.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `sounddevice` can't find a device | `python scripts/demo_stream.py --list-devices`, then pass `--device N` |
| No audio input | Check the USB mic is picked up: `arecord -l`. Use the correct card |
| Model never fires | Threshold 0.9 is tuned to the research checkpoint; try `--threshold 0.7` |
| False triggers on pure tones | Known research-artifact behavior (multilingual Run 7); it is not a production engine |
| PyTorch too heavy | Use the ONNX path (Section 3) — ONNX Runtime is much lighter than torch on ARM |

## Honest boundaries

- **Not production quality.** Deployed wake words run single-digit FRR at a
  false-accept rate quoted per hour; this model is a research artifact.
- **Multilingual is open.** English is the only language with meaningful data.
- **Latency unmeasured on real hardware.** The ~0.5 ms/frame figure is desktop
  CPU; a Pi will differ (likely still comfortably real-time at 100 frames/s).

---

*O1-Sound · MIT · part of the [AwareLiquid](https://awareliquid.ai) O-Series research line.*
