#!/usr/bin/env python
"""Evaluate a trained wake-word model: FRR at a fixed false-accept budget.

    python eval.py --ckpt checkpoints/o1sound.pt --root data/mswc --split test

Accuracy is the wrong headline for a wake word. What matters on a device is the
false-reject rate at a false-accept rate the user will tolerate -- conventionally
quoted per hour of speech. This script sweeps the decision threshold, picks the
operating point that meets `--target-far`, and reports FRR per language there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
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
from o1sound.keywords import GREETINGS


def collect_scores(model, frontend, loader, device):
    """Wake probability plus binary label and language for every clip.

    In multi-class mode the wake score is the total probability mass on the
    greeting classes (1..N), i.e. the OR the device actually acts on -- not the
    probability of naming the right language. Labels collapse to binary for the
    same reason.
    """
    model.eval()
    scores, labels, langs = [], [], []
    with torch.no_grad():
        for wav, label, lang in loader:
            probs = torch.softmax(model(frontend(wav.to(device))), dim=1)
            p = probs[:, 1:].sum(dim=1) if probs.shape[1] > 2 else probs[:, 1]
            scores.append(p.cpu())
            labels.append((label > 0).long())
            langs.extend(lang)
    return torch.cat(scores), torch.cat(labels), langs


def sweep(scores, labels, target_far):
    """Lowest threshold whose false-accept rate still meets the budget."""
    best = None
    for thr in torch.linspace(0.01, 0.99, 99):
        pred = (scores >= thr).long()
        neg = labels == 0
        pos = labels == 1
        far = (pred[neg] == 1).float().mean().item() if neg.any() else 0.0
        frr = (pred[pos] == 0).float().mean().item() if pos.any() else 1.0
        if far <= target_far and (best is None or frr < best[2]):
            best = (float(thr), far, frr)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--out", default="")
    ap.add_argument("--confusable-frac", type=float, default=0.0,
                    help="fraction of eval negatives drawn from words nearest the "
                         "wake word by edit distance. A model trained on hard "
                         "negatives and scored on uniform ones is measured on a "
                         "distribution shift, not on its quality — match this to "
                         "training, or report both")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = O1Sound(O1SoundConfig(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    frontend = LogMel().to(device)

    # Rebuild the dataset the way the checkpoint was trained: n_classes > 2
    # means the greeting labels must be restored, or every positive collapses
    # to class 1 and the per-language table becomes meaningless.
    multiclass = int(ck["config"].get("n_classes", 2)) > 2
    ds = MSWCWakeWord(args.root,
                      KeywordSpec(GREETINGS, confusable_frac=args.confusable_frac),
                      args.split, multiclass=multiclass)
    if args.confusable_frac > 0:
        print(f"eval negatives: {args.confusable_frac:.0%} confusable")
    if multiclass:
        print(f"multi-class checkpoint: {ck['config']['n_classes']} classes, "
              f"wake score = sum over greeting classes")
    dl = DataLoader(ds, args.batch, shuffle=False, collate_fn=collate)
    scores, labels, langs = collect_scores(model, frontend, dl, device)

    picked = sweep(scores, labels, args.target_far)
    if picked is None:
        print(f"no threshold reaches FAR <= {args.target_far:.3f} on this set.")
        return 1
    thr, far, frr = picked
    print(f"operating point: threshold {thr:.2f}  FAR {far:.4f}  FRR {frr:.4f}")

    per_lang = {}
    for lg in sorted(set(langs)):
        m = torch.tensor([x == lg for x in langs])
        pred = (scores[m] >= thr).long()
        lab = labels[m]
        npos, nneg = int((lab == 1).sum()), int((lab == 0).sum())
        per_lang[lg] = {
            "frr": float((pred[lab == 1] == 0).float().mean()) if npos else None,
            "far": float((pred[lab == 0] == 1).float().mean()) if nneg else None,
            "n_pos": npos, "n_neg": nneg,
        }

    print(f"\n{'lang':<8}{'FRR':>8}{'FAR':>8}{'n_pos':>8}{'n_neg':>8}")
    for lg, r in sorted(per_lang.items(), key=lambda kv: -(kv[1]["frr"] or 0)):
        f = "  n/a" if r["frr"] is None else f"{r['frr']:>8.3f}"
        a = "  n/a" if r["far"] is None else f"{r['far']:>8.3f}"
        print(f"{lg:<8}{f}{a}{r['n_pos']:>8}{r['n_neg']:>8}")

    worst = max((v["frr"] or 0.0 for v in per_lang.values()), default=0.0)
    print(f"\nworst-language FRR {worst:.3f} — this is the number that bounds any "
          f"multilingual claim, not the mean.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"threshold": thr, "far": far, "frr": frr,
                        "per_language": per_lang, "split": args.split}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
