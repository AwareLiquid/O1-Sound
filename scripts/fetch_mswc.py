#!/usr/bin/env python
"""Download per-language MSWC keyword subsets.

    python scripts/fetch_mswc.py --languages en,de,fr,es --out data/mswc

MSWC is distributed per language, which is the only reason training on twenty
languages is tractable on one machine: you never need the full corpus, only the
greeting directory plus enough other words to form negatives.

The archives are large. This script downloads and extracts them one language at
a time, skips any language already present, and never deletes anything.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/public-datasets-mswc/audio"


SPLITS_BASE = "https://storage.googleapis.com/public-datasets-mswc/splits"


def fetch(lang: str, out: Path, force: bool) -> bool:
    dest = out / lang
    if dest.exists() and not force:
        print(f"  {lang}: already present, skipped")
        return True
    url = f"{BASE}/{lang}.tar.gz"
    tmp = out / f"{lang}.tar.gz"
    out.mkdir(parents=True, exist_ok=True)
    print(f"  {lang}: downloading {url}")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:
        print(f"  {lang}: FAILED ({exc}). Check the code against the MSWC index; "
              f"not every locale uses a bare two-letter code (e.g. sv-SE).")
        return False
    print(f"  {lang}: extracting")
    # Windows reserves device names (nul, con, prn, aux, com1-9, lpt1-9) at
    # EVERY path depth, and MSWC word folders are real words -- Dutch has a
    # word directory literally named "nul" (zero), which kills extractall
    # partway through on Windows. Skip such members with a tally: losing one
    # negative word is harmless, dying mid-archive is not.
    reserved = ({"nul", "con", "prn", "aux"}
                | {f"com{i}" for i in range(1, 10)}
                | {f"lpt{i}" for i in range(1, 10)})
    skipped_reserved = 0
    with tarfile.open(tmp) as tf:
        members = []
        for m in tf.getmembers():
            parts = [p.split(".")[0].lower()
                     for p in m.name.replace("\\", "/").split("/")]
            if any(p in reserved for p in parts):
                skipped_reserved += 1
                continue
            members.append(m)
        tf.extractall(out, members=members, filter="data")
    if skipped_reserved:
        print(f"  {lang}: skipped {skipped_reserved} member(s) under "
              f"Windows-reserved names")
    tmp.unlink(missing_ok=True)

    # The audio archive contains ONLY clips/ -- the split assignments ship as a
    # separate small archive (~hundreds of KB). Without it the loader falls
    # back to "everything is train" and dev/test are silently empty.
    stmp = out / f"{lang}_splits.tar.gz"
    try:
        urllib.request.urlretrieve(f"{SPLITS_BASE}/{lang}.tar.gz", stmp)
        with tarfile.open(stmp) as tf:
            for m in tf.getmembers():
                if m.name.endswith("_splits.csv"):
                    m.name = Path(m.name).name
                    tf.extract(m, dest, filter="data")
        print(f"  {lang}: splits csv installed")
    except Exception as exc:
        print(f"  {lang}: WARNING no splits archive ({exc}) -- loader will "
              f"treat every clip as train and dev/test will be empty")
    finally:
        stmp.unlink(missing_ok=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", required=True, help="comma-separated MSWC codes")
    ap.add_argument("--out", default="data/mswc")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    ok = [l for l in langs if fetch(l, out, args.force)]
    print(f"\n{len(ok)}/{len(langs)} languages available under {out}")
    if len(ok) != len(langs):
        print("Train only claims the languages that actually downloaded.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
