"""Multilingual Spoken Words Corpus (MSWC) loader for wake-word training.

MSWC ships one archive per language, laid out as

    {lang}/clips/{word}/{hash}.opus
    {lang}/{lang}_splits.csv

`scripts/fetch_mswc.py` downloads and extracts the per-language keyword subsets;
this module only reads what is already on disk, so training never depends on the
network.

Decoding: .wav is read with the stdlib; .opus/.mp3 need `soundfile` (libsndfile
>= 1.2 handles opus). If a clip cannot be decoded it is skipped and counted --
silently dropping a whole language because one codec is missing would be much
worse than a loud tally at the end.
"""

from __future__ import annotations

import csv
import random
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TARGET_SR = 16000


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        width = w.getsampwidth()
        if width != 2:
            raise ValueError(f"expected 16-bit PCM, got {width * 8}-bit: {path}")
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
    return audio, sr


def _read_any(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix.lower() == ".wav":
        return _read_wav(path)
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            f"{path.suffix} needs `soundfile` (pip install soundfile). "
            "Only .wav is readable without it."
        ) from exc
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear resampling. Adequate here: MSWC is already 16 kHz or 48 kHz, and
    wake-word energy lives well below the range where a better kernel matters."""
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    if n_out <= 1:
        return np.zeros(1, dtype=np.float32)
    idx = np.linspace(0, len(x) - 1, n_out, dtype=np.float32)
    return np.interp(idx, np.arange(len(x), dtype=np.float32), x).astype(np.float32)


def fit_length(x: np.ndarray, n: int, rng: random.Random | None = None) -> np.ndarray:
    """Centre-pad or randomly crop to exactly `n` samples."""
    if len(x) == n:
        return x
    if len(x) > n:
        start = rng.randint(0, len(x) - n) if rng else (len(x) - n) // 2
        return x[start : start + n]
    pad = n - len(x)
    left = rng.randint(0, pad) if rng else pad // 2
    return np.pad(x, (left, pad - left))


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein on the written form -- a cheap stand-in for phonetic
    similarity that needs no pronunciation dictionary and works in every
    script MSWC ships."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class KeywordSpec:
    """Which word counts as the wake word, per language.

    `positives` maps language code -> the word directory that is the greeting in
    that language. Everything else sampled from that language becomes negative
    material, which is what teaches the model to reject phonetic neighbours
    rather than just 'any speech'.
    """

    positives: dict[str, str]
    negatives_per_language: int = 400
    # Uniformly-sampled negatives never test the boundary where it matters: a
    # detector only has to beat "hello" against "seven" to look fine, and then
    # fires on "hollow" in the field. Reserve this fraction of the negative
    # budget for the words closest to the wake word by edit distance.
    confusable_frac: float = 0.0
    confusable_max_distance: int = 3
    field_names: list[str] = field(default_factory=lambda: ["LINK", "WORD", "SPLIT"])


@dataclass
class Example:
    path: Path
    label: int          # 1 = wake word, 0 = negative
    language: str


class MSWCWakeWord(Dataset):
    """Binary wake-word dataset assembled across languages.

    The label is binary on purpose: the deployed decision is "did the greeting
    occur", not "which language was it". Language is kept on each example so the
    evaluator can report per-language false-reject rates, which is the number
    that actually decides whether the multilingual claim holds.
    """

    def __init__(
        self,
        root: str | Path,
        spec: KeywordSpec,
        split: str = "train",
        window_s: float = 1.0,
        sample_rate: int = TARGET_SR,
        seed: int = 0,
        multiclass: bool = False,
        augment=None,
    ) -> None:
        self.root = Path(root)
        self.spec = spec
        self.split = split
        self.sample_rate = sample_rate
        self.n_samples = int(window_s * sample_rate)
        self.rng = random.Random(seed)
        self.skipped: dict[str, int] = {}
        self.missing_wake: dict[str, str] = {}
        # Multi-class mode gives every greeting its own class and lumps all
        # non-greetings into class 0. Run 1 showed why binary fails: one class
        # spanning hello, cześć, γεια and سلام has more variance inside it than
        # between it and the negatives, so there is no coherent target to learn.
        # The wake decision at inference is an OR over classes 1..N.
        self.multiclass = multiclass
        self.augment = augment
        self.confusables: dict[str, list[str]] = {}
        self.confusable_noop: dict[str, bool] = {}
        self.class_names: list[str] = ["_other_"]
        self.lang_to_class: dict[str, int] = {}
        if multiclass:
            for lang in spec.positives:
                self.lang_to_class[lang] = len(self.class_names)
                self.class_names.append(f"{lang}:{spec.positives[lang]}")

        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.root} not found. Run scripts/fetch_mswc.py first."
            )

        self.examples: list[Example] = []
        for lang, word in spec.positives.items():
            self.examples.extend(self._collect_language(lang, word))
        if not self.examples:
            raise RuntimeError(
                f"no clips found under {self.root} for split={split!r}. "
                "Check the language codes in the keyword spec against the "
                "directories that were actually downloaded."
            )
        self.rng.shuffle(self.examples)

    def _split_of(self, lang: str) -> dict[str, str]:
        """clip filename -> split, from the language's splits csv (if present)."""
        csv_path = self.root / lang / f"{lang}_splits.csv"
        if not csv_path.exists():
            return {}
        mapping: dict[str, str] = {}
        with csv_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                link = row.get("LINK") or row.get("link")
                sp = (row.get("SET") or row.get("SPLIT") or "").upper()
                if link:
                    mapping[Path(link).name] = {"TRAIN": "train", "DEV": "dev", "TEST": "test"}.get(sp, "train")
        return mapping

    def _collect_language(self, lang: str, wake_word: str) -> list[Example]:
        clips = self.root / lang / "clips"
        if not clips.exists():
            self.skipped[lang] = -1  # language absent entirely
            return []
        split_map = self._split_of(lang)

        def keep(p: Path) -> bool:
            if not split_map:
                return True
            return split_map.get(p.name, "train") == self.split

        out: list[Example] = []
        pos_dir = clips / wake_word
        wake_label = self.lang_to_class.get(lang, 1) if self.multiclass else 1
        if pos_dir.exists():
            out += [Example(p, wake_label, lang) for p in sorted(pos_dir.iterdir()) if keep(p)]
        else:
            # The language is on disk but its wake folder is not -- a partial
            # extract, or a keyword string that does not match MSWC's native
            # spelling. Either way the language would silently contribute
            # negatives only and quietly vanish from the positive set, which
            # is exactly the failure that must not pass unnoticed.
            self.missing_wake[lang] = wake_word

        word_dirs = [d for d in sorted(clips.iterdir())
                     if d.is_dir() and d.name != wake_word]
        budget = self.spec.negatives_per_language
        n_conf = int(budget * self.spec.confusable_frac)
        chosen: list[Path] = []

        if n_conf > 0 and len(word_dirs) and budget >= sum(
                1 for d in word_dirs for p in d.iterdir() if keep(p)):
            # The budget already covers every available negative, so no
            # sampling happens and the confusable fraction cannot change the
            # composition. Silently returning the same set would make an
            # ablation look like a null result when it never ran.
            self.confusable_noop[lang] = True

        if n_conf > 0:
            scored = sorted(
                ((_edit_distance(d.name, wake_word), d) for d in word_dirs),
                key=lambda kv: kv[0],
            )
            near = [d for dist, d in scored if dist <= self.spec.confusable_max_distance]
            self.confusables[lang] = [d.name for d in near[:12]]
            pool: list[Path] = []
            for d in near:
                pool += [p for p in sorted(d.iterdir()) if keep(p)]
            self.rng.shuffle(pool)
            chosen += pool[:n_conf]

        rest: list[Path] = []
        for d in word_dirs:
            rest += [p for p in sorted(d.iterdir()) if keep(p)]
        self.rng.shuffle(rest)
        seen = set(chosen)
        chosen += [p for p in rest if p not in seen][: budget - len(chosen)]

        out += [Example(p, 0, lang) for p in chosen]
        return out

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:
        ex = self.examples[i]
        try:
            audio, sr = _read_any(ex.path)
        except Exception:
            self.skipped[ex.language] = self.skipped.get(ex.language, 0) + 1
            audio, sr = np.zeros(self.n_samples, dtype=np.float32), self.sample_rate
        audio = _resample_linear(audio, sr, self.sample_rate)
        rng = self.rng if self.split == "train" else None
        audio = fit_length(audio, self.n_samples, rng)
        if self.augment is not None:
            audio = self.augment(audio)
        return torch.from_numpy(np.ascontiguousarray(audio)), ex.label, ex.language

    def label_balance(self) -> tuple[int, int]:
        """(wake, non-wake). In multi-class mode any class > 0 is a wake word."""
        pos = sum(1 for e in self.examples if e.label > 0)
        return pos, len(self.examples) - pos

    @property
    def n_classes(self) -> int:
        return len(self.class_names) if self.multiclass else 2

    def languages(self) -> list[str]:
        return sorted({e.language for e in self.examples})


def collate(batch):
    wavs, labels, langs = zip(*batch)
    return torch.stack(wavs), torch.tensor(labels, dtype=torch.long), list(langs)
