# O1-Sound results

Chronological. Negative results stay in.

---

## Run 1 — 2026-07-31 — multilingual greeting, 9 languages, CPU

**Verdict: the pipeline works end to end. The model does not. Do not ship a
number from this run.**

### Setup

| | |
|---|---|
| Config | `hidden=640, layers=2` — 1,258,322 params, 5.03 MB fp32, 5,120 B carried state |
| Data | MSWC, 9 languages actually on disk (cs, el, id, nl, pl, pt, ro, sv-SE, uk) |
| Train | 1,441 clips — **91 wake** / 1,350 negative |
| Test | 1,372 clips — **22 wake** / 1,350 negative |
| Schedule | 25 epochs, batch 64, AdamW lr 3e-3 cosine, class-weighted CE (weight ≈ 14.8) |
| Hardware | CPU, 24 threads, ~29 s/epoch |

### Result

At the operating point that meets a 5% false-accept budget:

```
threshold 0.75    FAR 0.0486    FRR 0.9091
```

Per language:

| lang | FRR | FAR | n_pos | n_neg |
|---|---:|---:|---:|---:|
| id | **1.000** | 0.093 | 2 | 400 |
| nl | **1.000** | 0.045 | 1 | 400 |
| pl | **1.000** | 0.040 | 3 | 400 |
| ro | **1.000** | 0.055 | 2 | 400 |
| sv-SE | **1.000** | 0.038 | 5 | 400 |
| uk | **1.000** | 0.052 | 1 | 400 |
| cs | 0.800 | 0.055 | 5 | 400 |
| pt | 0.667 | 0.038 | 3 | 400 |
| el | n/a | 0.023 | 0 | 400 |

**The model misses 91% of wake words and never fires at all for six of the nine
languages.** Best dev accuracy across training was 0.8463 — against a
**0.9840 baseline for a model that simply never fires**. Predicting "no wake
word" unconditionally scores better than what trained. Loss plateaued at ~0.48
by epoch 20 and dev accuracy oscillated between 0.65 and 0.85 throughout, never
trending.

### Why — and why it is not (only) a data-volume problem

The obvious cause is sample count: 91 positives spread over 9 languages is 1–5
positives per language in the test split, which cannot support a claim either
way. More data is necessary.

But the framing deserves scrutiny before more GPU time is spent on it. A
conventional keyword spotter learns **one** fixed acoustic pattern. This task
as posed — "a greeting in any language" — asks a single binary class to cover
`hello ∪ cześć ∪ γεια ∪ سلام ∪ вітаю ∪ …`, phrases that share no phonetic
structure. That is a 17-way OR, not a wake word, and it plausibly needs both
far more data *per language* and an architecture that classifies which greeting
before deciding whether one occurred.

### Next, in order

1. **Add the two large languages.** en (301 wake clips) and fa (562) would take
   the positive set from 91 to ~950, roughly 10×. Cost: 35 GB + 6.8 GB of
   downloads.
2. **Sanity-check the framing on one language first.** Train en alone as a
   single-keyword spotter. If a 1.3 M-parameter model cannot learn "hello"
   from 301 clips, the problem is not multilinguality and no amount of extra
   languages will help.
3. Only if (2) succeeds does the multilingual claim become worth measuring
   again — and it is bounded by the worst language, not the mean.

### What this run does establish

The harness is real: MSWC download (audio + the separately-distributed splits
archive), opus decode, official train/dev/test assignment, class-weighted
training, threshold sweep to a fixed FAR budget, per-language FRR, and ONNX
export at 5.03 MB with 3.7e-09 parity. Six genuine defects surfaced only by
running it — wrong native-script keywords, a missing splits archive, Windows
reserved device names, silent missing-wake languages, a console-encoding crash,
and a missing output directory. Those are fixed and pinned.

Artefacts: `results/real_v1.json`.
