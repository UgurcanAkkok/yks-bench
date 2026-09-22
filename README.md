# yks-bench

Benchmarking **decision models** on the 2026 Turkish university entrance exam (YKS).

593 five-option questions, professionally written distractors, gold answers from
ÖSYM's own key. The exam was sat on 20–21 June 2026 and published afterwards, so for
locally-run encoders pretrained before that date it is an **uncontaminated** test set.

Two models, one harness, one base-URL apart.

| | accuracy | 95% CI |
| --- | --- | --- |
| **`jev-1.13.0`** (hosted TypeSafe) | **82.3%** | [79.0, 85.2] |
| `laya` 0.3.5 `english` (local) | 27.0% | [23.6, 30.7] |
| `laya` 0.3.5 `typed-decisions` | 25.1% | [21.8, 28.8] |
| `laya` 0.3.5 `multilingual` | 22.3% | [19.1, 25.8] |
| majority-class baseline | 24.6% | — |
| random | 20.0% | — |

The baseline that matters is **24.6%**, not 20% — the gold letters are not uniform.
No local configuration clears it, across three checkpoints, three phrasings and five
option orders. Full numbers and method in **[RESULTS.md](RESULTS.md)**.

## What this measures

Not just accuracy. The interesting question for a model that returns *calibrated
decisions* is whether the score comes from reading, and whether the confidence means
anything. So every run carries controls:

| control | Jev | local `english` |
| --- | --- | --- |
| delete the question, keep the options | **−32.5pp** [−36.9, −28.2] | +2.9 [−1.0, +6.7] |
| swap in a different item's passage | **−41.5pp** [−46.2, −36.8] | −0.3 [−4.0, +3.5] |
| remove the passage | −6.1pp [−9.2, −3.3] | −0.2 [−4.0, +3.3] |

Jev's score responds to what it is given. The local model's does not — every interval
contains zero, which is the signature of a model that is not reading the state.

The honest caveat: Jev still scores **49.7% with the question deleted entirely**, so
about half of these questions are answerable from the option set alone. The
incremental value of reading is ~33pp, not the full 62pp over chance.

**Confidence gating works.** ECE 0.025, fitted temperature 0.95 (already calibrated):

| coverage | 25% | 50% | 75% | 100% |
| --- | --- | --- | --- | --- |
| accuracy | **100.0%** | 98.6% | 94.8% | 82.3% |

## Section breakdown

`jev-1.13.0`, V5 phrasing, **all 691 printed questions** — every test at its full
official length, figures included and unseen. TYT and AYT each have a test called
Fen Bilimleri; they are different tests and are kept apart.

| test | accuracy | 95% CI | n | of which have a figure |
| --- | --- | --- | --- | --- |
| YDT Almanca (German) | 92.5% | [84.6, 96.5] | 80 | 0 |
| YDT Fransızca (French) | 92.5% | [84.6, 96.5] | 80 | 0 |
| YDT Rusça (Russian) | 88.8% | [80.0, 94.0] | 80 | 0 |
| YDT İngilizce (English) | 87.5% | [78.5, 93.1] | 80 | 0 |
| TYT Sosyal Bilimler | 80.0% | [60.9, 91.1] | 25 | 3 |
| AYT Sosyal Bilimler-2 | 78.3% | [64.4, 87.7] | 46 | 6 |
| TYT Türkçe | 77.5% | [62.5, 87.7] | 40 | 0 |
| AYT Türk Dili + Sosyal-1 | 75.0% | [59.8, 85.8] | 40 | 4 |
| YDT Arapça (Arabic) | 71.2% | [60.5, 80.0] | 80 | 0 |
| AYT Fen Bilimleri | 60.0% | [44.6, 73.7] | 40 | 18 |
| TYT Fen Bilimleri | 60.0% | [38.7, 78.1] | 20 | 10 |
| **AYT Matematik** | **17.5%** | [8.7, 32.0] | 40 | 26 |
| **TYT Temel Matematik** | **17.5%** | [8.7, 32.0] | 40 | 25 |

The ordering is almost exactly the figure column in reverse. Language tests carry no
figures at all and run 87–93%; both mathematics tests are roughly two-thirds figures
and land at 17.5%, below the 20% a random guesser would score. Science sits between,
with about half its questions carrying a figure.

That is two effects at once — no figure, and no arithmetic — and §4.2 of
[RESULTS.md](RESULTS.md) separates them: **on maths it can actually read, it scores
55.7%**. The text-only version of this table (593 questions) is in RESULTS §3.6.

Local `laya english` for comparison, text-only: İNG 45.0%, TEM 42.9%, ALM 36.2%, and
everything else between 16% and 28%. Only İNG clears the baseline, and with twelve
sections tested roughly one such result is expected by chance.

> **Run-to-run variation.** Hosted Jev is not deterministic. Two runs over the same
> 593 items with identical requests disagreed on 19 answers (3.2%), max |Δp| 0.27,
> with the flips concentrated on near-ties (p ≈ 0.23–0.50). Accuracy moved 0.3pp
> (82.3% → 82.6%), so no conclusion turns on it, but a single-run figure carries a
> small wobble on top of its sampling interval.

## What it would score on the actual exam

The table above uses the 593 questions answerable from text alone. A candidate
doesn't get to skip the other 98, so the exam score runs **all 691 printed
questions**, with the figures unseen. Overall accuracy across all 691: **74.2%**.

ÖSYM scores in nets, not accuracy: `net = correct − wrong/4`, blanks free. The
quarter-mark penalty is set so that random guessing is worth exactly zero.

| exam | test | net | of | | exam | test | net | of |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TYT | Türkçe | 28.75 | 40 | | AYT | Türk Dili – Sos-1 | 27.50 | 40 |
| TYT | Sosyal Bilimler | 13.75 | 20 | | AYT | Sosyal Bilimler-2 | 28.75 | 40 |
| TYT | Temel Matematik | **−1.25** | 40 | | AYT | Matematik | **−1.25** | 40 |
| TYT | Fen Bilimleri | 10.00 | 20 | | AYT | Fen Bilimleri | 20.00 | 40 |
| | **TYT total** | **51.25** | 120 | | YDT | İngilizce | **67.50** | 80 |
| | | | | | | **AYT total** | **75.00** | 160 |

**Both mathematics tests come out negative.** At −1.25 net it would have scored
higher by leaving the whole test blank, and lower than a candidate who filled the
sheet at random. English at 67.50 / 80 is its strongest result.

No puan is quoted: converting nets to a YKS score standardises against that year's
candidate population, and those constants aren't published in usable form. A
placement score also folds in a school GPA, which a model doesn't have.

### The figure penalty, and a calibration trap

| | accuracy | n | | | accuracy | n |
| --- | --- | --- | --- | --- | --- | --- |
| text only | 82.0% | 599 | | verbal, text only | 84.9% | 538 |
| has a figure | 23.9% | 92 | | quantitative, text only | 55.7% | 61 |

A figure costs ~30pp whatever the subject, and quantitative material is
*additionally* weaker. "Maths is at chance" was partly an artifact of maths being
where the figures are — on maths it can read, it scores 55.7%.

The more useful finding is what confidence does there:

| group | n | accuracy | mean top-p | gap |
| --- | --- | --- | --- | --- |
| verbal, text only | 538 | 84.9% | 82.6% | +2.4 |
| quantitative, text only | 61 | 55.7% | 63.4% | −7.7 |
| **quantitative, has a figure** | 79 | 20.3% | 50.0% | **−29.7** |
| all | 691 | 74.2% | 76.5% | −2.2 |

The overall gap of −2.2pp is excellent and hides the subgroup that matters. The
model can't see that a figure *exists*, so it reads the surrounding text as though
nothing were missing and reports ordinary confidence about an answer it had no
basis to give. **Missing information the model cannot detect does not lower its
confidence.** A confidence gate catches none of it: answering pays whenever
`p > 0.2`, and the top probability never falls that low on any of the 691, so
optimal blanking produces zero blanks.

Check calibration inside the subgroup you intend to gate on, not just overall.

```bash
python exam_score.py results/jev-1.13.0-all-691.jsonl --variant V5
```

## Method in brief

1. **Extract.** `extract_yks.py` reads the ÖSYM PDFs. The booklets carry a 45°
   watermark drawn glyph by glyph; every watermark line has a rotated writing
   direction, so keeping only `dir == (1, 0)` removes it cleanly. Question numbers
   sit alone in their own text block at the column's left edge, which distinguishes
   them from numbered items inside a stimulus. All 691 printed questions come out
   with five options and a gold answer; 593 are answerable from text alone.
2. **Ask.** One `choice` question per item. Criteria are keyed by the option text
   with `null` descriptions — the documented form for value selection. Keying by
   `A`–`E` puts a meaningless letter token in front of every option, because laya
   renders an option as `"<key>: <description>"`.
3. **Control.** choices-only, shuffled-state, no-passage, five cyclic option
   rotations, Choice vs five Nouls, and batched-vs-solo for shared-passage groups.
4. **Report.** Wilson intervals everywhere; paired bootstrap for any comparison;
   ECE on the top probability (not on `confidence`, which for a 5-option Choice is
   just `(5·p_max − 1)/4`); risk–coverage for the gating claim.

Phrasings are **reported, never selected** — for a model sold on "write a question,
get a calibrated answer", sensitivity to phrasing is a result, not a hyperparameter.
There is no dev/test split except for fitting the calibration temperature.

## Files

```
extract_yks.py     ÖSYM PDF  →  JSONL (watermark removal, two-column layout, answer key)
bench_yks.py       run a model over the set; seven arms; writes full distributions
bench_report.py    all metrics, recomputed from disk; never calls a model
bench_probe.py     diagnose an at-chance score: is the model broken, or is the task out of scope?
exam_score.py      results  →  YKS nets, figure penalty, per-subgroup calibration
data/              structural metadata per question (id, section, gold, has_figure) -- no text
results/           13,426 rows (Jev) + 68,154 rows (local); ids, labels, probabilities
RESULTS.md         full results, method, limitations, and the harness bugs found
```

## Running it

```bash
pip install pymupdf transformers        # extraction only
```

Fetch the booklets from ÖSYM into `data/yks2026/` — [2026-YKS question booklets and
answer keys](https://www.osym.gov.tr/2026yks-tyt-ayt-ve-ydt-temel-soru-kitapciklari-ve-cevap-anahtarlari),
then:

```bash
python extract_yks.py data/yks2026/*.pdf --out questions.jsonl              # all 691
python extract_yks.py data/yks2026/*.pdf --text-only --out questions_textonly.jsonl

# hosted
python bench_yks.py --out results/jev-1.13.0.jsonl \
    --checkpoints jev-latest --variants V5,V4 --rotations 5 \
    --workers 6 --url https://api.typesafe.ai      # key from .env or $TYPESAFE_API_KEY

# local (unbatched: concurrent micro-batching moves probabilities by up to 0.043 in
# bf16, wider than the top-1 margin on a quarter of these items)
python serve.py --offline --preload all \
    --max-len 2048 --head-max-len 768 --option-max-tokens 128 --no-batching
python bench_yks.py --out results/laya-local-bare.jsonl --variants V1 --rotations 5

python bench_report.py results/jev-1.13.0.jsonl --sections-for jev-latest

# all 691 printed questions, for the exam score
python bench_yks.py --questions questions.jsonl --out results/jev-1.13.0-all-691.jsonl \
    --checkpoints jev-latest --variants V5 --rotations 1 --skip-controls \
    --workers 6 --url https://api.typesafe.ai
python exam_score.py results/jev-1.13.0-all-691.jsonl --variant V5
```

The hosted runs are 13,426 rows for **$0.09**.

## A note on the harness

Two defects were found *after* the first full run, and both required redoing it. The
answer letters (§7.1 in RESULTS.md) reversed the local checkpoint ranking; the
question-inside-the-state (§7.2) changed nothing but contradicted the docs. Both are
recorded rather than quietly fixed, because an at-chance result is exactly the
situation where a harness bug is easiest to mistake for a finding — and `bench_probe.py`
exists for that reason: it tells you whether a floor score means "broken" or "out of
scope". Here it says the local model is sound (10/10 on tasks it was built for) and
the exam is simply beyond it.

## Provenance

Questions © ÖSYM. Reproduction and redistribution are not permitted, so **the PDFs
and the extracted question text are not in this repository** — only the extractor and
the results, which contain ids, labels and probabilities but no question text.
`extract_yks.py` rebuilds the dataset from the published booklets.

Benchmark code is MIT.
