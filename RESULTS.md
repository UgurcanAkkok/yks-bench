# YKS 2026 — benchmark results

A decision model against a national university entrance exam it could not have seen.

**Date of run:** 22–23 September 2026
**Models:** `jev-1.13.0` (hosted) and `laya` 0.3.5 checkpoints `english`, `multilingual`,
`typed-decisions` (local, RTX 4070 Ti SUPER, bf16)
**Items:** 593 five-option questions from 2026-YKS, sat 20–21 June 2026
**Rows collected:** 80,198 across both models
**Cost:** $0.056 for the hosted run (1.32M input tokens at $0.042/Mtok)

---

## 1. Headline

| | accuracy | 95% CI | beats majority (24.6%)? |
| --- | --- | --- | --- |
| **`jev-1.13.0` hosted** | **82.3%** | [79.0, 85.2] | **yes** |
| `laya english` local | 27.0% | [23.6, 30.7] | no |
| `laya typed-decisions` local | 25.1% | [21.8, 28.8] | no |
| `laya multilingual` local | 22.3% | [19.1, 25.8] | no |

Random guessing scores 20%. The gold letters are not uniformly distributed, so
always answering the commonest one scores 24.6% — that, not 20%, is the baseline a
result has to clear. **No local configuration clears it under any phrasing, any
option order, or any checkpoint.** Nine local configurations were tried.

The two models ran through the same code with one base-URL change.

---

## 2. Method

### 2.1 Dataset

ÖSYM publishes the 2026-YKS booklets with answer keys after the exam.
`extract_yks.py` turns the PDFs into JSONL. Two layout facts do the work:

* The booklets carry a 45° watermark drawn glyph by glyph. Every watermark line has
  a rotated writing direction, so keeping only lines whose `dir` is `(1, 0)` removes
  it completely.
* A question number sits alone in its own text block in a 10pt band at the left edge
  of its column. Numbered items inside a stimulus (`1. Gün: …`) and the section intro
  (`1. Bu testte 40 soru vardır.`) carry text on the same line, so they never look
  like a question start. A monotonic `n → n+1` guard backs this up.

| booklet | questions | with 5 options | with gold | text-only |
| --- | --- | --- | --- | --- |
| TYT (Türkçe / Sosyal / Temel Mat. / Fen) | 125 | 125 | 125 | 86 |
| AYT (TDE-SB1 / SB2 / Matematik / Fen) | 166 | 166 | 166 | 107 |
| YDT (İNG / ALM / FRA / RUS / AR) | 400 | 400 | 400 | 400 |
| **total** | **691** | **691** | **691** | **593** |

Every extracted question has five options and a gold answer, and the per-section
counts match the printed booklets exactly.

Two independent flags are recorded. `has_figure` means a bitmap sits inside the
question, so the text is missing a table, graph or diagram. `complete` means the
stem and all five options came out as text — six maths questions have no bitmap but
options drawn as vector fractions, which a bitmap check alone would miss. The
**593-item benchmark set requires both**, which is why the quantitative sections are
thin (MAT 12, TEM 14): most of them lose a figure.

### 2.2 Contamination

The exam was sat 20–21 June 2026 and published afterwards. Both local encoders
(ModernBERT-large, mmBERT-base) were pretrained well before that, so for the local
model this is a genuinely uncontaminated test set. For hosted `jev-1.13` no such
claim can be established either way, and none is made here.

### 2.3 Question construction

Each item becomes one `choice` question. Five phrasings were tried and **all are
reported — none is selected**. For a model whose promise is "write a question, get a
calibrated answer", sensitivity to phrasing is a result, not a hyperparameter, so
there is no dev/test split. (A split is used in exactly one place: fitting the
calibration temperature, which fits a real parameter.)

| variant | state | instructions |
| --- | --- | --- |
| V1 | stem + passage | generic Turkish |
| V2 | `{soru, parça}` | generic, referencing `soru` |
| V3 | `{soru, parça}` | the exam's question sentence |
| V4 | passage only | the exam's question sentence |
| V5 | passage only | the exam's question, framed in English |

V4 is the form the docs prescribe — *"the state contains the content and supporting
facts; questions define the judgments"*. V5 additionally frames the task in English,
which laya already does implicitly by wrapping instructions in `"choice question: "`.

### 2.4 Option encoding

laya renders a choice option as `"<key>: <description>"`. Criteria keyed `A`–`E`
therefore put an information-free letter token in front of every option. The
documented form for value-selection keys criteria **by the candidate text** with
`null` descriptions, which is what this benchmark uses. See §7.

### 2.5 Controls

| arm | what it does | what it tests |
| --- | --- | --- |
| choices-only | five options, empty state | is the option set alone sufficient? |
| shuffled | question paired with another item's state | does the state matter? |
| no-passage | passage removed from passage-based items | does the passage matter? |
| rotations | five cyclic rotations of the options | is the answer a function of content? |
| nouls | five independent yes/no questions per item | which decomposition is better? |
| together / alone | a shared-passage group batched vs one at a time | are parallel questions independent? |

Controls use the same phrasing as the main arm, so a paired difference isolates one
change rather than two.

### 2.6 Statistics

Accuracy carries a Wilson interval everywhere. Two configurations are compared only
with a paired bootstrap (5,000 draws) over the items both answered. At n=593 a 95%
interval is roughly ±3.5pp; at n=12 it is ±25pp, which is why the quantitative
sections are reported but not leaned on.

Calibration is computed on the top probability, not on `confidence`: for a 5-option
Choice, `confidence = (5·p_max − 1) / 4`, a monotone rescaling that carries no
independent information. ECE uses ten equal-mass bins.

### 2.7 Execution

The local server ran **unbatched**. Concurrent micro-batching shifts probabilities
by up to 0.043 in bf16, and a quarter of items have a top-1 margin narrower than
that (§6). The hosted run used six concurrent workers, where our own batching cannot
perturb anything.

---

## 3. Results — hosted `jev-1.13.0`

### 3.1 Controls

| arm | accuracy | 95% CI | paired vs real question |
| --- | --- | --- | --- |
| real question | 82.3% | [79.0, 85.2] | — |
| choices-only | 49.7% | [45.7, 53.8] | **−32.5** [−36.9, −28.2] |
| shuffled | 40.8% | [36.9, 44.8] | **−41.5** [−46.2, −36.8] |
| no-passage | 79.8% | [75.7, 83.3] | −6.1 [−9.2, −3.3] |

Deleting the question costs 32.5pp and giving the wrong passage costs 41.5pp. Those
are the responses of a model that is reading.

**But choices-only is 49.7%, far above chance.** Roughly half of these questions are
substantially answerable from the option set alone — the artifact effect Balepur et
al. (ACL 2024) describe. The honest reading of the headline is that the *incremental*
contribution of reading the question is ~33pp, not the full 62pp over chance.

Removing the passage costs only 6.1pp, which fits: most YKS stems are self-contained
and the `directions` passage is often redundant.

### 3.2 Option order

| variant | mean acc | spread | all 5 agree | vote acc | picked position 1–5 |
| --- | --- | --- | --- | --- | --- |
| V4 | 82.9% | 2.0 | 85.3% | 83.6% | 19.7 / 20.2 / 19.2 / 20.2 / 20.6 |
| V5 | 82.6% | 1.7 | 84.5% | 83.6% | 19.6 / 20.3 / 19.1 / 20.0 / 21.0 |

Position selection is essentially uniform. Order is a non-issue for Jev.

### 3.3 Choice vs five Nouls

| variant | choice | nouls | paired difference | mean Σ nouls |
| --- | --- | --- | --- | --- |
| V4 | 82.0% | 70.3% | **−11.6** [−15.2, −7.9] | 1.910 |
| V5 | 82.3% | 69.1% | **−13.2** [−16.7, −9.6] | 1.908 |

One Choice clearly beats five Nouls. The Noul sums average 1.91, nowhere near 1,
exactly as the jaggedness page says to expect.

### 3.4 Parallel questions

9 argmax flips in 426 (2.1%), max |Δp| 0.19, mean |Δp| 0.008, with a byte-identical
state and only the question count changing. Small, but a real departure from exact
independence.

### 3.5 Calibration and confidence gating

| ECE | Brier | fitted T | ECE after T | acc @25% | @50% | @75% | @100% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.025 | 0.238 | 0.95 | 0.042 | **100.0%** | 98.6% | 94.8% | 82.3% |

The fitted temperature is 0.95 — already calibrated, nothing to fix. The
risk–coverage curve is what the product claim promises: keep the most confident
quarter and every answer is right.

### 3.6 Section breakdown (V5, rotation 0)

| section | test | accuracy | 95% CI | n |
| --- | --- | --- | --- | --- |
| ALM | YDT German | 92.5% | [84.6, 96.5] | 80 |
| SOS | TYT Social Sciences | 90.9% | [72.2, 97.5] | 22 |
| FRA | YDT French | 90.0% | [81.5, 94.8] | 80 |
| RUS | YDT Russian | 87.5% | [78.5, 93.1] | 80 |
| İNG | YDT English | 87.5% | [78.5, 93.1] | 80 |
| SB2 | AYT Social Sciences 2 | 84.6% | [70.3, 92.8] | 39 |
| FEN | TYT/AYT Science | 80.0% | [62.7, 90.5] | 30 |
| TDE-SB1 | AYT Literature + Soc. 1 | 77.8% | [61.9, 88.3] | 36 |
| TÜR | TYT Turkish | 77.5% | [62.5, 87.7] | 40 |
| AR | YDT Arabic | 71.2% | [60.5, 80.0] | 80 |
| **MAT** | **AYT Mathematics** | **50.0%** | [25.4, 74.6] | 12 |
| **TEM** | **TYT Basic Mathematics** | **21.4%** | [7.6, 47.6] | 14 |

Verbal sections run high-80s to low-90s. **Basic mathematics is at chance.** This is
jaggedness item #2 — *"Jev is not a calculator, keep arithmetic in code"* — measured
on a real exam. Note the small n on both maths sections; the intervals are wide.

---

## 4. Results — local `laya` 0.3.5

### 4.1 Accuracy

All nine configurations (3 checkpoints × 3 phrasings) land between 20.6% and 27.0%.
Under the corrected option encoding: english 27.0% [23.6, 30.7], typed-decisions
25.1% [21.8, 28.8], multilingual 22.3% [19.1, 25.8].

### 4.2 Controls — nothing moves

| checkpoint | choices-only | no-passage | shuffled |
| --- | --- | --- | --- |
| english | +2.9 [−1.0, +6.7] | −0.2 [−4.0, +3.3] | −0.3 [−4.0, +3.5] |
| multilingual | +1.5 [−2.4, +5.6] | −4.5 [−8.9, +0.0] | +3.7 [−0.3, +7.9] |
| typed-decisions | +1.9 [−2.4, +6.1] | −3.1 [−6.8, +0.7] | +0.3 [−3.7, +4.2] |

Every interval contains zero. Deleting the question, removing the passage, or
substituting the wrong passage entirely all cost nothing measurable. This is the
signature of a model that is not reading the state.

### 4.3 Option order

| checkpoint | mean acc | spread | all 5 agree | picked position 1–5 |
| --- | --- | --- | --- | --- |
| english | 25.3% | 4.4 | 22.1% | **36.1** / 18.6 / 15.5 / 14.0 / 15.8 |
| multilingual | 23.7% | 4.0 | 31.0% | 16.9 / 22.5 / 21.6 / 20.4 / 18.6 |
| typed-decisions | 23.4% | 3.0 | 27.3% | **29.4** / 18.3 / 16.3 / 15.8 / 20.3 |

Two checkpoints show a marked first-slot preference, and on roughly three quarters
of items the winner changes when the options are merely rotated.

### 4.4 Choice vs Nouls, and parallel questions

No decomposition helps: english −3.4 [−8.1, +1.2], multilingual +2.4 [−2.4, +6.9],
typed-decisions −1.3 [−5.7, +3.0]. Noul sums 2.84–3.03.

Independence: 5–13 flips in 426 with max |Δp| 0.010–0.040 — the same magnitude as
the pure bf16 batch-shape noise measured in §6, so most likely numerics.

### 4.5 Calibration

| checkpoint | ECE | Brier | fitted T | ECE after T | acc @25% | @100% |
| --- | --- | --- | --- | --- | --- | --- |
| english | 0.070 | 0.799 | 2.10 | 0.097 | 33.1% | 27.0% |
| multilingual | 0.244 | 0.917 | 18.30 | 0.034 | 24.3% | 22.3% |
| typed-decisions | 0.055 | 0.793 | 1.65 | 0.092 | 33.8% | 25.1% |

`multilingual` ships with no fitted temperatures and its ECE of 0.244 shows it;
refitting takes it to 0.034. But good calibration at chance only means the model
correctly reports that it does not know.

### 4.6 Section breakdown (`english`, V1, rotation 0)

| section | accuracy | 95% CI | n |
| --- | --- | --- | --- |
| İNG | 45.0% | [34.6, 55.9] | 80 |
| TEM | 42.9% | [21.4, 67.4] | 14 |
| ALM | 36.2% | [26.6, 47.2] | 80 |
| FRA | 27.5% | [18.9, 38.1] | 80 |
| FEN | 26.7% | [14.2, 44.4] | 30 |
| SOS | 22.7% | [10.1, 43.4] | 22 |
| RUS | 21.2% | [13.7, 31.4] | 80 |
| SB2 | 20.5% | [10.8, 35.5] | 39 |
| TDE-SB1 | 19.4% | [9.8, 35.0] | 36 |
| TÜR | 17.5% | [8.7, 32.0] | 40 |
| MAT | 16.7% | [4.7, 44.8] | 12 |
| AR | 16.2% | [9.7, 25.8] | 80 |

İNG at 45.0% is the only section whose interval clears the baseline. With twelve
sections tested, roughly one such result is expected by chance.

---

## 5. Diagnostics — is the local model broken?

No. `bench_probe.py` separates "broken" from "out of scope".

**Sanity, 10/10.** Routing, sentiment, category, world knowledge and reading
comprehension, in English and Turkish, with sensible confidence — `iade` p=1.00,
`elma` p=1.00, "Paris" only p=0.35 (honest: that is recall, not reading). Re-run with
meaningful keys, with `A`–`E` keys and with bare options: 10/10, 10/10, 9/10.

**Cloze ladder, 6/7.** English absurd-distractor, same-part-of-speech, collocation and
discourse-connective all correct (`However`, p=0.78). Turkish absurd and same-POS
correct at p=0.87 / 0.96; Turkish discourse connective wrong.

**Graded distractors, same exam states.** Real options 25.0% → four concrete nouns
37.1% (multilingual) / **57.8%** (english). The model does extract signal from an exam
passage; it cannot resolve fine distinctions.

**Hypotheses tested and rejected** as explanations for the local result: distractor
quality (real / same-section / off-topic all ≈25%), state length (no trend from <200
to 800+ chars), answer extractability (19.6% when the gold text appears verbatim in
the state vs 25.4% when it does not), lexical overlap (the model agrees with the
most-overlapping option only 22.7% of the time), and option length (mild shortest-
option preference, 26.5% vs 20%). Only negation showed an effect: 19.6% on
"hangisi **söylenemez**" items vs 25.4% otherwise, n=46.

---

## 6. Phase 0 — preconditions

**Determinism: exact.** Two sequential runs, max |Δp| = 0.00000000.

**Batching: not invariant.** Sequential vs 16 concurrent requests gives max |Δp| =
0.0426 with one argmax flip in 40. The server docstring claims <0.01; that does not
hold on this workload. **10 of 40 items have a top-1 margin narrower than that
drift**, so ~2.5% of answers would flip on scheduling timing alone. Hence unbatched.

**Truncation: zero**, asserted item by item across 593 items × 3 variants × 2
tokenizers, at `--max-len 2048 --head-max-len 768 --option-max-tokens 128`.

---

## 7. Harness defects found and fixed

Both were found after the first full run and both required re-running.

**1. Answer letters inside the sequence.** `render_options` emits `"<key>: <description>"`,
so criteria keyed `A`–`E` put a meaningless letter token in front of every option.
This also invalidated a design argument — that the LLM position-bias literature could
not apply because laya has no answer-letter tokens. The harness had put them there.

Fixing it moved results ±4pp, **reversed the local checkpoint ranking** (english
23.1→27.0, multilingual 25.0→22.3) and improved rotation agreement from ~18% to
22–31%. It did not change any verdict. `--option-encoding` selects between the two;
`bare` is the default and matches the documented form.

**2. The question was inside the state.** V1 — the configuration behind the first
run's headline numbers, controls and calibration — put the exam's interrogative
sentence in the state and sent a generic instruction, contradicting *"separate content
from questions"*. Fixed as V4. No effect on results (23.9 / 25.5 / 23.4%).

**3. Controls were phrased differently from the main arm.** The controls were pinned
to V1 while the hosted main arm ran V5, so a paired difference would have mixed "the
state changed" with "the question changed". Controls now inherit the main arm's
phrasing, and the hosted controls were re-run.

Also fixed while building: `--option-max-tokens` reached only the batched code path,
so `--no-batching` and the OOM fallback silently ignored it; and `--resume` checked
"already recorded" *after* the HTTP call, so resuming re-issued every request and
discarded the answers.

Checked against the docs and found correct: `serialize_state` is plain `json.dumps`;
255 options are supported; multiple questions per call; string instructions.

---

## 8. Limitations

* **Choices-only at 49.7% for Jev** means the headline overstates reading ability.
  The passage-independent component is large.
* **Small n on the quantitative sections** (MAT 12, TEM 14). The figure filter removes
  most maths questions because their content is a bitmap. The maths conclusion is
  directionally clear but the intervals are wide.
* **One exam, one year, one language.** Nothing here generalises to other languages
  or to Jev's performance on the decision tasks it is actually sold for.
* **Twelve sections tested.** Expect roughly one spurious "clears the baseline" result.
* **No hosted contamination claim.** Only the local result is provably uncontaminated.
* The hosted run used concurrency; the local run did not. This is deliberate, but the
  two are not identical execution regimes.

---

## 9. Reproducing

```bash
# 1. fetch the booklets (URLs in README.md) into data/yks2026/
# 2. extract
python extract_yks.py data/yks2026/*.pdf --text-only --out questions_textonly.jsonl

# 3. local, unbatched, no truncation
python serve.py --offline --preload all \
    --max-len 2048 --head-max-len 768 --option-max-tokens 128 --no-batching
python bench_yks.py --out results/laya-local-bare.jsonl --variants V1 --rotations 5

# 4. hosted
python bench_yks.py --out results/jev-1.13.0.jsonl \
    --checkpoints jev-latest --variants V5,V4 --rotations 5 \
    --workers 6 --url https://api.typesafe.ai

# 5. report, and diagnose an at-chance result
python bench_report.py results/jev-1.13.0.jsonl --sections-for jev-latest
python bench_probe.py sanity ladder options state
```

`results/*.jsonl` carry the full probability distribution for every item, so every
number above can be recomputed without re-running inference.

---

## 10. Provenance

Questions © ÖSYM, published at
<https://www.osym.gov.tr/2026yks-tyt-ayt-ve-ydt-temel-soru-kitapciklari-ve-cevap-anahtarlari>.
Reproduction and redistribution are not permitted, so the PDFs and the extracted
question text are **not** in this repository — only the extractor and the results,
which contain ids, labels and probabilities but no question text.
