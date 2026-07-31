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

---

## Run 2 — 2026-08-01 — English only, the narrow test

**Verdict: the architecture learns a wake word. Run 1's failure was the task
definition, not the model.**

Run 1 left one question that had to be answered before spending anything more:
can a 1.3 M-parameter liquid core learn ONE keyword at all? Same code, same
config, same hardware — only the label set changed, from "a greeting in any of
9 languages" to "the English word hello".

### Setup

Identical to Run 1 except the data: `hidden=640, layers=2`, 30 epochs, batch 64,
AdamW lr 3e-3 cosine, class-weighted CE, CPU. MSWC English, fetched with
`--stream` (301 wake clips landed in ~3 GB instead of the archive's 35 GB).

| split | clips | wake | negative | never-fire baseline |
|---|---:|---:|---:|---:|
| train | 1,418 | 219 | 1,199 | 0.8456 |
| dev | 186 | 41 | 145 | 0.7796 |
| test | 195 | 41 | 154 | 0.7897 |

### Result

Test split, at the operating point meeting a 5% false-accept budget:

```
threshold 0.82    FAR 0.0455    FRR 0.1463
```

> **Correction (same day).** This was first published as *FRR 0.2439 at
> threshold 0.78*. That figure is wrong: it was measured while the English
> archive was **still downloading**, so the test split was incomplete. On the
> finished data the point it names does not exist — at threshold 0.78 the true
> FAR is 0.0519, which fails the 5% budget. Re-running the pre-refactor code on
> the complete data reproduces **0.1463** exactly, confirming the refactor did
> not move the number and the original run was the faulty one. The lesson is
> procedural, not numerical: never evaluate against a directory another process
> is still writing.

Best dev accuracy 0.9194 against a 0.7796 never-fire baseline — the first
configuration in this repository that beats "always say no".

Against Run 1 at the same false-accept budget:

| run | positives | FRR @ FAR≈5% | vs never-fire |
|---|---:|---:|---|
| Run 1 — 9 languages | 91 | 0.909 | worse (0.846 vs 0.984) |
| **Run 2 — English only** | **301** | **0.146** | **better (0.919 vs 0.780)** |

**Miss rate falls to roughly one third.** Two variables moved — 3.3× the
positives and a single acoustic target — and this run does not separate them.
What it does settle is the question it was built to answer: the architecture is
not the blocker.

### What this is NOT

- **Not production quality.** Deployed wake words run FRR in the low single
  digits at a false-accept rate quoted per hour of audio, not per clip. 15%
  missed activations is a bad user experience.
- **Not a multilingual result.** It is one word in one language. The
  multilingual claim remains unsupported, and Run 1 is evidence against it at
  this data scale.
- **Dev flatters it.** Dev FRR was 0.073 against test 0.146 on 41 positives
  each. With samples this small the gap is partly threshold selection on dev,
  and the honest number is the test one.

### Next

The label set is the lever. Run 1 asked one binary class to span `hello ∪ cześć
∪ γεια ∪ سلام ∪ вітаю` — phrases sharing no phonetic structure, so within-class
variance exceeded between-class variance. Run 2 removed that and the model
worked. The fix is therefore not "more languages into the same class":

1. **Multi-class head, OR at inference.** 17 greeting classes plus "other";
   the wake signal is an OR over the greeting logits. Each class stays
   acoustically coherent, per-language diagnosis becomes possible, and the cost
   is ~10 K parameters — the 5.03 MB budget is unaffected.
2. **Augmentation.** No augmentation exists yet; class weighting only tells the
   loss to care more about the same recordings. Time shift, noise at varied
   SNR, speed perturbation and SpecAugment are standard for exactly this
   small-positive regime.
3. **Confusable negatives.** Negatives are sampled uniformly at random.
   Wake-word training needs near-misses (hello vs hollow, yellow, hell) or the
   decision boundary is never tested where it matters.
4. **Pooling.** `forward()` mean-pools over time, diluting a keyword that
   occupies a fraction of the window. Max or attention pooling suits both the
   task and the multi-timescale core better.

Artefacts: `results/narrow_en.json`, `ck/narrow_en.metrics.json`.
