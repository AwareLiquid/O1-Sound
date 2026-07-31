#!/usr/bin/env python
"""Train the O1-Sound wake-word spotter on MSWC.

    python train.py --root data/mswc --epochs 20 --out checkpoints/o1sound.pt

Reports per-language false-reject rate every epoch, because a multilingual claim
stands or falls on the worst language, not the mean.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys

# MSWC keywords are native script by design (cześć, привет, سلام), and this
# script prints them. A Windows console defaults to a legacy codepage (GBK,
# CP1252) that cannot encode them, so an ordinary progress line would raise
# UnicodeEncodeError and kill a training run mid-flight. Degrade the glyph,
# never the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from o1sound import LogMel, O1Sound, O1SoundConfig
from o1sound.data import KeywordSpec, MSWCWakeWord, collate
from o1sound.data.augment import WaveformAugment, spec_augment
from o1sound.keywords import GREETINGS


def evaluate(model, frontend, loader, device):
    """Returns overall accuracy plus per-language false-reject / false-accept."""
    model.eval()
    per_lang: dict[str, list[int]] = {}
    correct = total = 0
    with torch.no_grad():
        for wav, label, langs in loader:
            wav, label = wav.to(device), label.to(device)
            logits = model(frontend(wav))
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.numel()
            # The deployed decision is "did a greeting occur", so in multi-class
            # mode FRR/FAR are measured on the OR over classes 1..N, not on
            # whether the right language was identified.
            wake_true = (label > 0)
            wake_pred = (pred > 0)
            for i, lg in enumerate(langs):
                fr, fa, npos, nneg = per_lang.setdefault(lg, [0, 0, 0, 0])
                if wake_true[i]:
                    npos += 1
                    fr += int(not wake_pred[i])
                else:
                    nneg += 1
                    fa += int(wake_pred[i])
                per_lang[lg] = [fr, fa, npos, nneg]
    rates = {
        lg: {
            "frr": (fr / npos) if npos else None,
            "far": (fa / nneg) if nneg else None,
            "n_pos": npos,
            "n_neg": nneg,
        }
        for lg, (fr, fa, npos, nneg) in per_lang.items()
    }
    return (correct / total if total else 0.0), rates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="extracted MSWC directory")
    ap.add_argument("--out", default="checkpoints/o1sound.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=640)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--negatives", type=int, default=400)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--multiclass", action="store_true",
                    help="one class per greeting plus 'other'; the wake decision "
                         "is an OR over the greeting classes. Run 1 showed a "
                         "single binary class cannot span phonetically unrelated "
                         "greetings")
    ap.add_argument("--pooling", choices=["mean", "max", "attn"], default="mean")
    ap.add_argument("--augment", action="store_true",
                    help="waveform augmentation on the TRAIN split only")
    ap.add_argument("--spec-augment", action="store_true",
                    help="frequency/time masking on the mel batch")
    ap.add_argument("--confusable-frac", type=float, default=0.0,
                    help="fraction of the negative budget reserved for words "
                         "closest to the wake word by edit distance")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    spec = KeywordSpec(GREETINGS, negatives_per_language=args.negatives,
                       confusable_frac=args.confusable_frac)

    aug = WaveformAugment(seed=args.seed) if args.augment else None
    train_ds = MSWCWakeWord(args.root, spec, "train", args.window, seed=args.seed,
                            multiclass=args.multiclass, augment=aug)
    # dev and test never see augmentation -- that would measure the augmentation.
    dev_ds = MSWCWakeWord(args.root, spec, "dev", args.window, seed=args.seed + 1,
                          multiclass=args.multiclass)
    pos, neg = train_ds.label_balance()
    print(f"train {len(train_ds)} clips ({pos} wake / {neg} negative) "
          f"over {len(train_ds.languages())} languages; dev {len(dev_ds)}")
    missing = [lg for lg, n in train_ds.skipped.items() if n == -1]
    if missing:
        print(f"WARNING: {len(missing)} language(s) in the spec are not on disk: "
              f"{', '.join(missing)} — the multilingual claim covers only what trained.")
    if train_ds.missing_wake:
        detail = ", ".join(f"{lg}({w})" for lg, w in train_ds.missing_wake.items())
        print(f"WARNING: {len(train_ds.missing_wake)} language(s) present but with NO "
              f"wake-word folder: {detail}. They contribute negatives only — check for a "
              f"partial extract or a keyword that does not match MSWC's native spelling.")

    train_dl = DataLoader(train_ds, args.batch, shuffle=True, collate_fn=collate, drop_last=True)
    dev_dl = DataLoader(dev_ds, args.batch, shuffle=False, collate_fn=collate)

    if train_ds.confusables:
        for lg, words in train_ds.confusables.items():
            print(f"  {lg} confusable negatives: {', '.join(words[:8])}")

    frontend = LogMel().to(device)
    n_classes = train_ds.n_classes
    model = O1Sound(O1SoundConfig(hidden=args.hidden, n_layers=args.layers,
                                  n_classes=n_classes, pooling=args.pooling)).to(device)
    print(f"head: {n_classes} classes, pooling={args.pooling}, "
          f"augment={'on' if aug else 'off'}, spec_augment={'on' if args.spec_augment else 'off'}")
    n_params = model.num_parameters()
    print(f"model {n_params:,} params = {n_params * 4 / 1e6:.2f} MB fp32, "
          f"carried state {model.state_bytes()} B/stream")

    # Negatives outnumber positives heavily; weight the loss so the model cannot
    # win by never firing.
    if n_classes == 2:
        weight = torch.tensor([1.0, max(1.0, neg / max(pos, 1))], device=device)
    else:
        # Per-class inverse frequency: the greeting classes are individually
        # tiny (4-28 clips for the tail languages) and would otherwise be
        # drowned by "other".
        counts = [0] * n_classes
        for e in train_ds.examples:
            counts[e.label] += 1
        total_n = sum(counts)
        weight = torch.tensor(
            [total_n / (n_classes * c) if c else 1.0 for c in counts],
            device=device, dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for wav, label, _ in train_dl:
            wav, label = wav.to(device), label.to(device)
            mel = frontend(wav)
            if args.spec_augment:
                mel = spec_augment(mel)
            loss = crit(model(mel), label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        acc, rates = evaluate(model, frontend, dev_dl, device)
        worst = max(
            ((lg, r["frr"]) for lg, r in rates.items() if r["frr"] is not None),
            key=lambda kv: kv[1], default=("-", 0.0),
        )
        print(f"epoch {ep:>3}  loss {tot / max(nb,1):.4f}  dev acc {acc:.4f}  "
              f"worst-language FRR {worst[1]:.3f} ({worst[0]})  {time.time() - t0:.0f}s")
        if acc > best:
            best = acc
            torch.save(
                {"model": model.state_dict(),
                 "config": vars(model.config),
                 "class_names": train_ds.class_names,
                 "dev_acc": acc,
                 "per_language": rates,
                 "languages": train_ds.languages()},
                args.out,
            )
            Path(args.out).with_suffix(".metrics.json").write_text(
                json.dumps({"dev_acc": acc, "per_language": rates}, indent=2), encoding="utf-8"
            )
    print(f"best dev acc {best:.4f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
