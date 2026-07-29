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
    with tarfile.open(tmp) as tf:
        tf.extractall(out, filter="data")
    tmp.unlink(missing_ok=True)
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
