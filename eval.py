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

from o1sound import LogMel, O1Sound, O1SoundConfig
from o1sound.data import KeywordSpec, MSWCWakeWord, collate
from o1sound.keywords import GREETINGS


def collect_scores(model, frontend, loader, device):
    """Wake-word probability plus label and language for every clip."""
    model.eval()
    scores, labels, langs = [], [], []
    with torch.no_grad():
        for wav, label, lang in loader:
            p = torch.softmax(model(frontend(wav.to(device))), dim=1)[:, 1]
            scores.append(p.cpu())
            labels.append(label)
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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu")
    model = O1Sound(O1SoundConfig(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    frontend = LogMel().to(device)

    ds = MSWCWakeWord(args.root, KeywordSpec(GREETINGS), args.split)
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
        Path(args.out).write_text(
            json.dumps({"threshold": thr, "far": far, "frr": frr,
                        "per_language": per_lang, "split": args.split}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
