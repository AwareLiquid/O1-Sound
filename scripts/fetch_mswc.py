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

# This script prints native-script keywords (cześć, вітаю, سلام); a legacy
# console codepage would raise UnicodeEncodeError mid-download. Same fix as
# train.py / eval.py: degrade the glyph, never the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
import shutil
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


def fetch_streaming(lang: str, out: Path, wake: str, max_neg_words: int,
                    max_clips_per_word: int) -> bool:
    """Stream the archive and keep only what training needs.

    A full MSWC language is up to 35 GB on disk (English) to supply ~300 clips
    of one greeting. This decompresses the tarball as it downloads and writes
    out only the wake-word folder plus a bounded sample of other words, so peak
    disk is a few hundred MB regardless of archive size. Bandwidth is unchanged
    -- the whole stream is still read -- but a laptop can run it.
    """
    dest = out / lang
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{lang}.tar.gz"
    print(f"  {lang}: streaming {url} (keeping '{wake}' + <= {max_neg_words} other words)")

    kept_wake = 0
    per_word: dict[str, int] = {}
    reserved = ({"nul", "con", "prn", "aux"}
                | {f"com{i}" for i in range(1, 10)}
                | {f"lpt{i}" for i in range(1, 10)})
    try:
        with urllib.request.urlopen(url) as resp,              tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for m in tf:                      # streaming: never seeks backwards
                if not m.isfile():
                    continue
                parts = m.name.replace("\\", "/").split("/")
                if len(parts) < 3 or parts[1] != "clips":
                    continue
                word = parts[2]
                if word.lower() in reserved:
                    continue
                is_wake = word == wake
                if not is_wake:
                    if word not in per_word and len(per_word) >= max_neg_words:
                        continue
                    if per_word.get(word, 0) >= max_clips_per_word:
                        continue
                src = tf.extractfile(m)
                if src is None:
                    continue
                target = dest / "clips" / word / Path(m.name).name
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "wb") as fh:
                    shutil.copyfileobj(src, fh)
                if is_wake:
                    kept_wake += 1
                else:
                    per_word[word] = per_word.get(word, 0) + 1
    except Exception as exc:
        print(f"  {lang}: FAILED ({exc})")
        return False

    print(f"  {lang}: kept {kept_wake} '{wake}' clips + "
          f"{sum(per_word.values())} negatives across {len(per_word)} words")
    if kept_wake == 0:
        print(f"  {lang}: WARNING no '{wake}' clips found — check the spelling "
              f"against MSWC's native-script folder names")
    return _fetch_splits(lang, dest)


def _fetch_splits(lang: str, dest: Path) -> bool:
    stmp = dest.parent / f"{lang}_splits.tar.gz"
    try:
        urllib.request.urlretrieve(f"{SPLITS_BASE}/{lang}.tar.gz", stmp)
        with tarfile.open(stmp) as tf:
            for m in tf.getmembers():
                if m.name.endswith("_splits.csv"):
                    m.name = Path(m.name).name
                    tf.extract(m, dest, filter="data")
        print(f"  {lang}: splits csv installed")
        return True
    except Exception as exc:
        print(f"  {lang}: WARNING no splits archive ({exc})")
        return True
    finally:
        stmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", required=True, help="comma-separated MSWC codes")
    ap.add_argument("--out", default="data/mswc")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stream", action="store_true",
                    help="decompress on the fly and keep only the wake word plus "
                         "a sample of negatives (peak disk stays in the hundreds "
                         "of MB even for a 35 GB language)")
    ap.add_argument("--max-neg-words", type=int, default=400,
                    help="with --stream, how many distinct non-wake words to keep")
    ap.add_argument("--max-clips-per-word", type=int, default=3,
                    help="with --stream, clips kept per negative word")
    args = ap.parse_args()

    out = Path(args.out)
    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    if args.stream:
        from o1sound.keywords import GREETINGS
        ok = []
        for l in langs:
            if l not in GREETINGS:
                print(f"  {l}: no verified greeting in keywords.py, skipped")
                continue
            if fetch_streaming(l, out, GREETINGS[l], args.max_neg_words,
                               args.max_clips_per_word):
                ok.append(l)
    else:
        ok = [l for l in langs if fetch(l, out, args.force)]
    print(f"\n{len(ok)}/{len(langs)} languages available under {out}")
    if len(ok) != len(langs):
        print("Train only claims the languages that actually downloaded.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
