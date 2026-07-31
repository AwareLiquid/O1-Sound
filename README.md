# O1-Sound

**A multilingual greeting wake-word spotter built on the O-Series liquid core.**

Always-on keyword spotting on a battery: the model listens continuously, holds a
fixed-size state no matter how long the microphone has been open, and fires when
it hears a greeting — "hello", "hola", "bonjour", "你好" — in any of the
languages it was trained on.

```
mic → log-mel (fixed, no weights) → liquid core (multi-timescale τ) → wake / not-wake
                                     ↑ carried state: 5,120 bytes, constant
```

## Honest status — what is and isn't validated

This section comes first on purpose.

**Validated (runs today, in CI)**
- 10 tests pass: streaming `step()` is numerically identical to the batched
  `forward()`, carried state stays constant over 500 frames, τ spans the
  configured range, the export CLI fits its budget, and the budget gate really
  fails the build when it cannot be met.
- **ONNX export: 5.03 MB fp32, 1.27 MB int8**, max |onnx − torch| = **3.7e-09**.
  Measured, reproducible with `python export_onnx.py --int8`.
- **Carried state: 5,120 bytes per stream**, independent of stream length — this
  is a property of the recurrence, verified by test, not a measurement that
  could drift.

**Measured — the architecture works, the multilingual framing does not (yet)**
- **Run 2** (2026-08-01, English only, 301 wake clips): **FRR 0.146 at FAR
  0.046** on held-out test. Best dev accuracy 0.919 against a 0.780 never-fire
  baseline — the first configuration here that beats "always say no".
- **Run 1** (2026-07-31, 9 languages, 91 wake clips): **FRR 0.909 at FAR
  0.049**, never firing at all for six of the nine. Dev 0.846 against a 0.984
  never-fire baseline — worse than saying no unconditionally.
- Same code, same config, same hardware; only the label set changed. **Miss
  rate fell to roughly one third**, which settles what Run 2 was built to
  settle: the architecture is not the blocker. Two variables moved (3.3× the
  positives, one acoustic target) and this pair of runs does not separate them.
- **Not production quality.** Deployed wake words run single-digit FRR at a
  false-accept rate quoted per hour, not per clip. And **the multilingual claim
  is still unsupported** — Run 1 is evidence against it at this data scale.
  The planned fix is a multi-class head OR'd at inference, not more languages
  poured into one binary class. Reasoning in [RESULTS.md](RESULTS.md).

**NOT validated — there is no shippable model in this repository**
- **No production-grade weights ship here.**
  `export_onnx.py` without `--ckpt` exports an *untrained* graph: the size is
  real, the behaviour is noise. It says so when you run it.
- The greeting list in `o1sound/keywords.py` covers 20 languages, but a language
  only counts once its MSWC directory has actually been downloaded and trained
  on. `train.py` prints a warning naming any language in the spec that is
  missing from disk, because "20 languages" and "20 languages we trained on" are
  different claims.
- **Latency is unmeasured on real hardware.** The Python `step()` loop runs at
  ~0.5 ms/frame on a desktop CPU, which says nothing about an MCU or a DSP. No
  power figure exists either.
- The 7 MB budget was a target given up front, not a constraint the architecture
  strained against — at hidden=640 it lands at 5.03 MB with room to spare.

**What would make the claims real**

```bash
python scripts/fetch_mswc.py --languages en,de,fr,es,it,pt,pl,ru,tr,id --out data/mswc
python train.py --root data/mswc --epochs 20 --out checkpoints/o1sound.pt
python eval.py  --ckpt checkpoints/o1sound.pt --root data/mswc --split test --out results/test.json
python export_onnx.py --ckpt checkpoints/o1sound.pt --out dist/o1sound.onnx --int8
```

`eval.py` reports **false-reject rate at a fixed false-accept budget, per
language**, and prints the worst language separately. That worst number — not
the mean, not accuracy — is what bounds any multilingual claim made about this
model.

## Why a liquid core

Each channel carries its own learnable time constant τ, parameterised as
`softplus(log_tau) + tau_min` and initialised geometrically across
10–240 ms. Short-τ channels track the current phoneme; long-τ channels hold the
envelope of the whole word. That spread is what separates a wake phrase from a
phonetically close neighbour without stacking depth — which matters when the
budget is a few megabytes and the model never sleeps.

Two consequences for the target device:

- **Streaming is O(1).** The state is a fixed `(batch, hidden)` tensor per layer.
  It does not grow with how long the microphone has been open, so there is no
  drift in memory or latency across a long session.
- **`step()` is what ships.** The exported ONNX graph is the single-frame step —
  one frame in, logits and the next state out — not a fixed-length window.

## Install

```bash
pip install -e .
# .opus clips additionally need: pip install soundfile
```

## Model sizes

Measured, `n_classes=2`:

| hidden | layers | params | fp32 | int8 | carried state |
|-------:|-------:|-------:|-----:|-----:|--------------:|
| 384 | 2 | 483,984 | 1.94 MB | 0.48 MB | 3,072 B |
| 512 | 2 | 841,872 | 3.37 MB | 0.84 MB | 4,096 B |
| **640** | **2** | **1,298,064** | **5.19 MB** | **1.30 MB** | **5,120 B** |
| 512 | 3 | 1,367,184 | 5.47 MB | 1.37 MB | 6,144 B |
| 768 | 2 | 1,852,560 | 7.41 MB | 1.85 MB | 6,144 B |

`hidden=640, layers=2` is the default: the largest 2-layer configuration that
still clears 7 MB in fp32 once ONNX graph overhead is counted (5.03 MB on disk).

## Layout

```
o1sound/features.py     log-mel frontend (torch.stft, no torchaudio, ONNX-clean)
o1sound/model.py        liquid core + streaming step()
o1sound/data/mswc.py    MSWC loader, per-language splits, binary wake/not-wake
o1sound/keywords.py     greeting word per language
train.py                training loop, per-language FRR each epoch
eval.py                 FRR at a fixed FAR budget, per language
export_onnx.py          streaming ONNX + int8 + hard size gate
```

## Related

- [everest-an/M1](https://github.com/everest-an/M1) — the MT-LNN / O-Series
  research line this core comes from.
- [awareliquid.ai](https://awareliquid.ai) — benchmarks and retractions.

## License

MIT.
