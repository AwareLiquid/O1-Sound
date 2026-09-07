"""Robust MSWC fetch: curl with resume/retry for the network, local
streaming extract for the disk (keep wake word + bounded negatives).

Replaces scripts/fetch_mswc.py --stream for flaky networks: the previous
path streams the tar through urllib, and a mid-stream network blip kills the
run (English needs a 34.8 GB read; we lost it twice that way). curl -C - 
resumes across failures, and the extraction then runs offline from the local
file. Peak disk = one tar (~35 GB worst case) + a few hundred MB of clips.

Usage:
    py -3.11 -X utf8 scripts/fetch_robust.py --languages en,de,fr --out data/mswc
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import shutil
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE = "https://storage.googleapis.com/public-datasets-mswc/audio"
SPLITS_BASE = "https://storage.googleapis.com/public-datasets-mswc/splits"
PROXY = "http://127.0.0.1:9674"
RESERVED = ({"nul", "con", "prn", "aux"}
            | {f"com{i}" for i in range(1, 10)}
            | {f"lpt{i}" for i in range(1, 10)})


def curl_download(url: str, dest: Path) -> bool:
    """Download with resume + retries; survives network blips."""
    cmd = ["curl", "-sS", "-x", PROXY, "-C", "-",
           "--retry", "30", "--retry-delay", "5", "--retry-all-errors",
           "-o", str(dest), url]
    p = subprocess.run(cmd)
    return p.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def stream_extract(tar_path: Path, dest: Path, wake: str,
                   max_neg_words: int, max_clips: int) -> int:
    """Offline streaming extract: wake word + bounded negative sample."""
    kept_wake = 0
    per_word: dict[str, int] = {}
    with tarfile.open(tar_path, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.replace("\\", "/").split("/")
            if len(parts) < 3 or parts[1] != "clips":
                continue
            word = parts[2]
            if word.lower() in RESERVED:
                continue
            is_wake = word == wake
            if not is_wake:
                if word not in per_word and len(per_word) >= max_neg_words:
                    continue
                if per_word.get(word, 0) >= max_clips:
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
    return kept_wake


def fetch_splits(lang: str, dest: Path) -> None:
    stmp = dest.parent / f"{lang}_splits.tar.gz"
    if curl_download(f"{SPLITS_BASE}/{lang}.tar.gz", stmp):
        try:
            with tarfile.open(stmp) as tf:
                for m in tf.getmembers():
                    if m.name.endswith("_splits.csv"):
                        m.name = Path(m.name).name
                        tf.extract(m, dest, filter="data")
            print(f"  {lang}: splits csv installed")
        except Exception as exc:
            print(f"  {lang}: WARNING splits extract failed ({exc})")
    else:
        print(f"  {lang}: WARNING no splits archive - dev/test will be empty")
    stmp.unlink(missing_ok=True)


def fetch_lang(lang: str, wake: str, out: Path,
               max_neg_words: int, max_clips: int) -> bool:
    dest = out / lang
    wake_dir = dest / "clips" / wake
    if wake_dir.exists() and any(wake_dir.iterdir()):
        print(f"  {lang}: wake clips already present, skipped")
        return True
    if dest.exists():
        print(f"  {lang}: partial (no '{wake}' clips), cleaning")
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    tmp = out / f"{lang}.tar.gz"
    print(f"  {lang}: downloading (curl resume) {BASE}/{lang}.tar.gz")
    if not curl_download(f"{BASE}/{lang}.tar.gz", tmp):
        print(f"  {lang}: FAILED download")
        return False
    print(f"  {lang}: extracting (wake='{wake}', <= {max_neg_words} neg words)")
    kept_wake = stream_extract(tmp, dest, wake, max_neg_words, max_clips)
    tmp.unlink(missing_ok=True)
    print(f"  {lang}: kept {kept_wake} '{wake}' clips")
    if kept_wake == 0:
        print(f"  {lang}: WARNING zero wake clips - check keyword spelling")
    fetch_splits(lang, dest)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", required=True)
    ap.add_argument("--out", default="data/mswc")
    ap.add_argument("--max-neg-words", type=int, default=400)
    ap.add_argument("--max-clips-per-word", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=3)
    args = ap.parse_args()

    from o1sound.keywords import GREETINGS

    out = Path(args.out)
    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    failed = []
    for lang in langs:
        if lang not in GREETINGS:
            print(f"  {lang}: no verified greeting in keywords.py, skipped")
            continue
        ok = False
        for attempt in range(1, args.attempts + 1):
            print(f"[{lang}] attempt {attempt}/{args.attempts}")
            if fetch_lang(lang, GREETINGS[lang], out,
                          args.max_neg_words, args.max_clips_per_word):
                ok = True
                break
        if not ok:
            failed.append(lang)
    print("done. failed:", failed if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
