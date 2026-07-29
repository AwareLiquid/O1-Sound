#!/usr/bin/env python
"""Export the streaming graph to ONNX and enforce the size budget.

    python export_onnx.py --ckpt checkpoints/o1sound.pt --out dist/o1sound.onnx --int8

Exports `step()` -- one frame in, logits plus the next state out -- because that
is what an always-on device actually runs. The batched `forward()` is a training
convenience; shipping it would bake in a fixed window length.

The size gate is a hard failure, not a warning: a wake-word model that quietly
grew past its budget is a broken deliverable, and the number is quoted publicly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

from o1sound import O1Sound, O1SoundConfig

BUDGET_MB = 7.0


class StreamingStep(nn.Module):
    """Wraps step() so the ONNX graph has flat tensor inputs and outputs."""

    def __init__(self, model: O1Sound) -> None:
        super().__init__()
        self.model = model
        self.n_layers = len(model.cells)

    def forward(self, mel_t: torch.Tensor, *state: torch.Tensor):
        logits, new_state = self.model.step(mel_t, list(state))
        return (logits, *new_state)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="trained checkpoint; omit to export an untrained graph")
    ap.add_argument("--out", default="dist/o1sound.onnx")
    ap.add_argument("--int8", action="store_true", help="also emit a dynamically quantised model")
    ap.add_argument("--hidden", type=int, default=640)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--budget-mb", type=float, default=BUDGET_MB)
    args = ap.parse_args()

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        cfg = O1SoundConfig(**ck["config"])
        model = O1Sound(cfg)
        model.load_state_dict(ck["model"])
        print(f"loaded {args.ckpt} (dev acc {ck.get('dev_acc', 'n/a')})")
    else:
        cfg = O1SoundConfig(hidden=args.hidden, n_layers=args.layers, n_classes=2)
        model = O1Sound(cfg)
        print("NO CHECKPOINT: exporting an UNTRAINED graph. Size is meaningful; "
              "accuracy is not. Do not publish accuracy from this artefact.")
    model.eval()

    wrapper = StreamingStep(model).eval()
    mel_t = torch.zeros(1, cfg.n_mels)
    state = model.init_state(1)
    out_names = ["logits"] + [f"state_out_{i}" for i in range(len(state))]
    in_names = ["mel_frame"] + [f"state_in_{i}" for i in range(len(state))]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (mel_t, *state),
        str(out),
        input_names=in_names,
        output_names=out_names,
        dynamic_axes={n: {0: "batch"} for n in in_names + out_names},
        opset_version=17,
        # The dynamo exporter still refuses parts of this graph; the legacy
        # tracer handles it and the step function has no data-dependent control
        # flow, so tracing is faithful here.
        dynamo=False,
    )
    size_mb = out.stat().st_size / 1e6
    print(f"{out}  {size_mb:.2f} MB")

    ok = size_mb <= args.budget_mb
    if args.int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            q = out.with_name(out.stem + ".int8.onnx")
            quantize_dynamic(str(out), str(q), weight_type=QuantType.QInt8)
            q_mb = q.stat().st_size / 1e6
            print(f"{q}  {q_mb:.2f} MB")
        except ImportError:
            print("onnxruntime not installed; skipped int8 (pip install onnxruntime)")

    # Numerical check against PyTorch -- an export that silently diverges is
    # worse than one that fails outright.
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        feeds = {in_names[0]: mel_t.numpy()}
        feeds.update({in_names[i + 1]: s.numpy() for i, s in enumerate(state)})
        onnx_logits = sess.run(None, feeds)[0]
        with torch.no_grad():
            torch_logits, _ = model.step(mel_t, state)
        drift = float(np.abs(onnx_logits - torch_logits.numpy()).max())
        print(f"max |onnx - torch| = {drift:.3e}")
        if drift >= 1e-4:
            print(f"FAIL: numerical drift {drift:.3e} exceeds 1e-4")
            ok = False
    except ImportError:
        print("onnxruntime not installed; skipped the numerical check")

    if not ok:
        print(f"FAIL: {size_mb:.2f} MB exceeds the {args.budget_mb:.1f} MB budget")
        return 1
    print(f"OK: within the {args.budget_mb:.1f} MB budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
