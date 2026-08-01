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

---

## Run 3 — 2026-08-01 — the four fixes, and why this test set cannot judge them

**Verdict: the fixes are implemented and behave as designed. This experiment
cannot tell whether they help. The blocker is now measurement, not modelling.**

Same data as run 2 (English, 301 wake clips), plus all four changes run 2
prescribed: waveform augmentation, SpecAugment, 35% confusable negatives, and
attention pooling.

| | run 2 | run 3 |
|---|---:|---:|
| final train loss | ~0.48 | **0.31** |
| dev FRR | 0.049 | 0.049 |
| dev FAR | 0.034 | 0.110 |
| **test FRR @ FAR 0.046** | **0.146** | **0.220** |

Test FRR went *up*. Before reading anything into that:

```
run 2:  6/41 missed   FRR 0.146   95% CI [0.038, 0.254]
run 3:  9/41 missed   FRR 0.220   95% CI [0.093, 0.346]
difference: 3 clips.  two-proportion z = 0.86 — not significant at 95%
```

**The whole difference is three clips out of forty-one.** One clip moves FRR by
0.024. The intervals overlap across most of their range, dev FRR is identical
at 0.049, and a 5-point FRR change is simply not resolvable at n=41. Neither
run is evidence for or against the fixes.

Two real observations survive the noise:

- **Train loss fell from ~0.48 to 0.31.** Augmentation is doing what
  augmentation does — the model has more to fit and fits it. That is an
  optimisation fact, independent of the test-set problem.
- **Dev FAR rose 0.034 → 0.110.** Expected and worth naming: run 3 trains
  against confusable negatives (hell, cell, alloy, apollo) but is *evaluated*
  against uniformly-sampled ones. A boundary tuned on near-misses is
  necessarily looser on easy negatives. Hard-negative training and easy-negative
  evaluation is a mismatch, and the evaluation set is the side that is wrong.

### Fixed here

Checkpoint selection used **dev accuracy**, which with negatives outnumbering
positives ~6:1 rewards a model drifting toward never firing. Both earlier runs
therefore saved on a criterion partly anti-correlated with the job. Selection is
now **balanced accuracy** over the wake decision — the mean of the per-class
rates, which a never-fire model cannot win.

### Next — fix the measurement first

1. **Enlarge the test set before running more configurations.** Persian has 562
   wake clips against English's 301; streaming it costs ~100 MB on disk. en+fa
   roughly triples the positives and makes a 5-point difference resolvable.
2. **Multiple seeds.** One run per configuration cannot separate a real effect
   from initialisation. Three seeds minimum before any config is called better.
3. **Evaluate against the negatives that were trained against.** Either put
   confusables in the eval set too, or report both — the current pairing
   measures a distribution shift as if it were model quality.
4. Only then is `--multiclass` worth running: the multilingual question is the
   expensive one, and it deserves an instrument that can read the answer.

Artefacts: `results/narrow_v3.json`, `ck/narrow_v3.metrics.json`.

---

## Run 4 — 2026-08-01 — en+fa, 3 seeds x 2 configs

**Verdict: the four fixes do not help. And seed variance alone is larger than
the difference between configurations, which retroactively invalidates runs 2
and 3 as comparisons.**

The instrument problem from run 3 was fixed first: Persian adds 562 wake clips
to English's 301 (streamed for 236 MB against a 6.8 GB archive), taking the
test split from 41 positives to 101. Then three seeds per configuration, 20
epochs each, everything else identical.

| config | FRR mean | sd | per-seed |
|---|---:|---:|---|
| baseline | **0.139** | 0.052 | 0.198, 0.119, 0.099 |
| + augment, SpecAugment, 35% confusables, attn pooling | **0.205** | 0.040 | 0.198, 0.247, 0.168 |

```
difference = +0.066 FRR (worse)
t(4) = 1.73 — not significant at 95%
```

The number that matters more than either mean:

```
seed-to-seed spread WITHIN the baseline:  0.099 FRR  (0.099 -> 0.198, ten clips)
difference BETWEEN the configurations:    0.066 FRR
```

**Initialisation moves the metric 1.5x more than the entire intervention
does.** Runs 2 and 3 were one seed each; the 0.146 vs 0.220 gap reported there
sits comfortably inside this noise band. Neither of those comparisons carried
information, and this run supersedes both.

Dev balanced accuracy was flat across all six runs (0.920-0.936), so the models
are of similar quality and most of the test-FRR spread is threshold selection on
101 positives, not a difference in what was learned.

### Two reasons the fixes may genuinely hurt, beyond noise

Worth stating because they are mechanisms, not excuses, and both are testable:

1. **Confusable negatives change the operating point, and the test set did not
   follow.** Training against hell/cell/alloy/apollo tightens the boundary
   where near-misses live; scoring against uniformly-sampled negatives then
   reads that tightening as a worse FRR at the same FAR. `eval.py` now takes
   `--confusable-frac` so both sides can be matched — this run did not use it.
2. **Equal epochs is unfair to augmentation.** Augmentation enlarges the
   effective dataset, so 20 epochs is fewer passes over the true data
   distribution than 20 epochs without it. The augmented runs may simply be
   less converged. A fair comparison equalises steps-to-plateau, not epochs.

### What this settles

- The measurement problem is real and was the binding constraint. It is now
  partly fixed (101 positives, 3 seeds) and partly not (variance still spans ten
  clips).
- **No configuration in this repository has been shown to beat any other.** The
  best supported statement remains run 2's: a single-keyword English detector
  reaches FRR ~0.14 at FAR ~0.05, with a seed-to-seed spread of roughly ±0.05.
- Any future claim needs 3+ seeds, matched negative distributions, and enough
  positives that one clip is not 1% of the metric.

Artefacts: `ck/enfa_{base,fix}_s{0,1,2}.pt`.
