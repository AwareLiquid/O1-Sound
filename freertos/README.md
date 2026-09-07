# freertos/ — the streaming C kernel

The O1-Sound model as plain C: log-mel frontend + layer norm + 2 liquid-core
cells + linear head, streaming one frame (480 samples) at a time with a
constant carried state (2 × HIDDEN floats).

```
mic → log-mel (480-pt DFT) → LayerNorm → liquid cell ×2 → head → wake / not-wake
                                  ↑ carried state: 5,120 B, constant
```

## Files

| File | Role |
|---|---|
| `o1sound_kernel.h/.c` | the kernel — reference implementation, parity-gated |
| `freertos_demo.c` | FreeRTOS task wiring (I2S ring buffer → step → debounced wake) |
| `generated/weights.h` | **generated** — do not commit. See below. |
| `../scripts/export_c_kernel.py` | checkpoint → `weights.h` (float32 arrays) |
| `../tests/test_c_parity.py` | host parity gate: C vs PyTorch, frame by frame |

## Workflow (host-side verification, no target hardware needed)

```bash
# 1. export weights from a checkpoint (research release or local training)
python scripts/export_c_kernel.py --ckpt checkpoints/o1sound.pt \
    --out freertos/generated/weights.h

# 2. build the kernel with any C compiler (zig cc works everywhere)
zig cc -O2 -Ifreertos freertos/o1sound_kernel.c tests/test_driver.c -lm -o /tmp/o1s

# 3. run the parity gate — C logits must match PyTorch within 1e-3 of the
#    logit span (the C frontend runs a float32 direct DFT where torch runs
#    an FFT, so the tolerance is relative, not absolute)
python tests/test_c_parity.py --ckpt checkpoints/o1sound.pt
#    → "PARITY PASS"
```

## Honest engineering notes

- The 480-point DFT here is a **reference implementation** (O(N²)). Correct and
  parity-gated, but the production MCU path is a mixed-radix FFT (or
  zero-padding to 512 with a re-validated filterbank). The kernel core
  (cells + head) is identical either way.
- `generated/weights.h` is a build artifact and is gitignored; regenerate it
  from the checkpoint you are actually deploying. It must match the checkpoint
  passed to `test_c_parity.py`.
- Edge frames: `torch.stft(center=True)` reflect-pads the first frames; the
  steady-state equivalence (last-480-samples window) holds for frames t ≥ 2.
  The parity test feeds the exact padded windows so ALL frames agree.
