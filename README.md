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

`jev-1.13.0`, V5 phrasing, rotation 0:

| section | test | accuracy | 95% CI | n |
| --- | --- | --- | --- | --- |
| ALM | YDT German | 92.5% | [84.6, 96.5] | 80 |
| SOS | TYT Social Sciences | 90.9% | [72.2, 97.5] | 22 |
| FRA | YDT French | 90.0% | [81.5, 94.8] | 80 |
| RUS | YDT Russian | 87.5% | [78.5, 93.1] | 80 |
| İNG | YDT English | 87.5% | [78.5, 93.1] | 80 |
| SB2 | AYT Social Sciences 2 | 84.6% | [70.3, 92.8] | 39 |
| FEN | TYT/AYT Science | 80.0% | [62.7, 90.5] | 30 |
| TDE-SB1 | AYT Literature + Social 1 | 77.8% | [61.9, 88.3] | 36 |
| TÜR | TYT Turkish | 77.5% | [62.5, 87.7] | 40 |
| AR | YDT Arabic | 71.2% | [60.5, 80.0] | 80 |
| **MAT** | **AYT Mathematics** | **50.0%** | [25.4, 74.6] | 12 |
| **TEM** | **TYT Basic Mathematics** | **21.4%** | [7.6, 47.6] | 14 |

Verbal reasoning runs high-80s to low-90s. **Basic mathematics is at chance** — a
direct measurement of the documented limitation that Jev is not a calculator and
arithmetic belongs in code. Both maths sections have small n (the figure filter
removes most maths questions, whose content is a bitmap), so the intervals are wide.

Local `english` for comparison: İNG 45.0%, TEM 42.9%, ALM 36.2%, then everything else
between 16% and 28%. Only İNG clears the baseline, and with twelve sections tested
roughly one such result is expected by chance.

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
results/           12,044 rows (Jev) + 68,154 rows (local); ids, labels, probabilities
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
```

The hosted run is 12,044 rows for **$0.056**.

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
