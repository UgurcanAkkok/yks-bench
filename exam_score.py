#!/usr/bin/env python3
"""Score a benchmark run the way ÖSYM scores a candidate.

Accuracy is not what a YKS result is measured in. A wrong answer costs a quarter
of a mark and a blank costs nothing, so the unit is the *net*:

    net = correct - wrong / 4

The penalty is set so that guessing at random is worth exactly zero in
expectation, which makes a net of 0.00 on a test a meaningful statement: the
model did no better than a candidate who filled the sheet at random.

This deliberately stops at nets. Converting nets to a YKS puan standardises each
test against that year's candidate population, and those constants are not
published in a usable form; a placement score also folds in a school GPA, which a
model does not have. Any puan printed here would be invented precision.

    python exam_score.py results/jev-1.13.0-all-691.jsonl

The run must cover all 691 printed questions, figures included -- a candidate
does not get to skip those. `--variant` and `--rotation` pick the arm to score.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Each test as ÖSYM counts it. Where a booklet prints two alternative groups --
# Din Kültürü and its substitute -- a candidate answers one, so the first is taken.
TESTS = [
    ("TYT", "Türkçe",               "2026-TYT", "TÜR",     40, 40),
    ("TYT", "Sosyal Bilimler",      "2026-TYT", "SOS",     20, 20),
    ("TYT", "Temel Matematik",      "2026-TYT", "TEM",     40, 40),
    ("TYT", "Fen Bilimleri",        "2026-TYT", "FEN",     20, 20),
    ("AYT", "Türk Dili ve Ed.-Sos-1", "2026-AYT", "TDE-SB1", 40, 40),
    ("AYT", "Sosyal Bilimler-2",    "2026-AYT", "SB2",     40, 40),
    ("AYT", "Matematik",            "2026-AYT", "MAT",     40, 40),
    ("AYT", "Fen Bilimleri",        "2026-AYT", "FEN",     40, 40),
    ("YDT", "İngilizce",            "2026-YDT", "İNG",     80, 80),
]

QUANTITATIVE = {"TEM", "MAT", "FEN"}


def wilson(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def load(path: Path, variant: Optional[str], rotation: int) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("arm") != "choice":
            continue
        if variant and row.get("variant") != variant:
            continue
        if row.get("rotation", 0) != rotation:
            continue
        out[row["id"]] = row
    return out


def top_probability(row: dict) -> float:
    if row.get("probabilities"):
        return max(row["probabilities"].values())
    return (4 * row["confidence"] + 1) / 5          # invert confidence = (5p-1)/4


def report_nets(answers: Dict[str, dict], meta: Dict[str, dict]) -> None:
    print(f"\n{'exam':<5}{'test':<26}{'correct':>8}{'wrong':>7}{'net':>9}{'of':>5}{'acc':>8}")
    print("-" * 68)
    totals: Dict[str, List[float]] = defaultdict(lambda: [0, 0, 0.0])
    missing = 0
    for exam, name, exam_id, section, answered, _printed in TESTS:
        ids = sorted((i for i, m in meta.items()
                      if m["exam"] == exam_id and m["section"] == section),
                     key=lambda i: meta[i]["number"])[:answered]
        seen = [i for i in ids if i in answers]
        missing += len(ids) - len(seen)
        correct = sum(1 for i in seen if answers[i]["correct"])
        wrong = len(seen) - correct
        net = correct - wrong / 4
        totals[exam][0] += correct
        totals[exam][1] += wrong
        totals[exam][2] += net
        print(f"{exam:<5}{name:<26}{correct:>8}{wrong:>7}{net:>9.2f}{len(seen):>5}"
              f"{100 * correct / max(1, len(seen)):>7.1f}%")
    print("-" * 68)
    for exam in ("TYT", "AYT", "YDT"):
        correct, wrong, net = totals[exam]
        print(f"{exam:<5}{'TOTAL':<26}{correct:>8}{wrong:>7}{net:>9.2f}{correct + wrong:>5}")
    if missing:
        print(f"\n  WARNING: {missing} questions of the official count are absent from this run.")


def report_blanking(answers: Dict[str, dict], meta: Dict[str, dict]) -> None:
    """Answering is worth it only when p(correct) > 0.2; below that a blank scores better."""
    answered = skipped = 0
    net_all = net_gated = 0.0
    for i, row in answers.items():
        hit = row["correct"]
        net_all += 1 if hit else -0.25
        if top_probability(row) > 0.2:
            answered += 1
            net_gated += 1 if hit else -0.25
        else:
            skipped += 1
    print(f"\nLeaving blanks when the model is unsure")
    print(f"  answer everything          net {net_all:8.2f}")
    print(f"  skip when top-p <= 0.2     net {net_gated:8.2f}   ({skipped} blanks, "
          f"{answered} answered)")
    if skipped == 0:
        print("  the model never reports a top probability at or below 0.2, so there is")
        print("  nothing to gate on -- see the calibration table below for why that matters.")


def report_calibration(answers: Dict[str, dict], meta: Dict[str, dict]) -> None:
    """Calibration inside each subgroup, not just overall.

    A good global ECE can hide a subgroup where the model is confidently wrong,
    and that subgroup is exactly the one a confidence gate is supposed to catch.
    """
    groups = [
        ("verbal, text only",       lambda m: m["section"] not in QUANTITATIVE and not m["has_figure"]),
        ("verbal, has a figure",    lambda m: m["section"] not in QUANTITATIVE and m["has_figure"]),
        ("quantitative, text only", lambda m: m["section"] in QUANTITATIVE and not m["has_figure"]),
        ("quantitative, has a figure", lambda m: m["section"] in QUANTITATIVE and m["has_figure"]),
        ("every question with a figure", lambda m: m["has_figure"]),
        ("all", lambda m: True),
    ]
    print(f"\n{'group':<32}{'n':>5}{'accuracy':>11}{'mean top-p':>12}{'gap':>8}")
    print("-" * 68)
    for label, keep in groups:
        rows = [answers[i] for i, m in meta.items() if i in answers and keep(m)]
        if not rows:
            continue
        acc = 100 * sum(1 for r in rows if r["correct"]) / len(rows)
        mean_p = 100 * sum(top_probability(r) for r in rows) / len(rows)
        print(f"{label:<32}{len(rows):>5}{acc:>10.1f}%{mean_p:>11.1f}%{acc - mean_p:>+8.1f}")
    print("\n  A negative gap is overconfidence: the model claims more than it delivers.")


def report_figures(answers: Dict[str, dict], meta: Dict[str, dict]) -> None:
    print(f"\n{'group':<32}{'accuracy':>10}{'95% CI':>16}{'n':>5}")
    print("-" * 68)
    for label, keep in (
        ("text only", lambda m: not m["has_figure"]),
        ("has a figure", lambda m: m["has_figure"]),
    ):
        rows = [answers[i] for i, m in meta.items() if i in answers and keep(m)]
        hits = sum(1 for r in rows if r["correct"])
        lo, hi = wilson(hits, len(rows))
        print(f"{label:<32}{100 * hits / len(rows):>9.1f}%  [{lo:5.1f},{hi:5.1f}]{len(rows):>5}")
    print("\n  A figure is a bitmap the model cannot see. These questions are still")
    print("  answered, because a candidate does not get to skip them.")


NAMES = {
    ("2026-TYT", "TÜR"): "TYT Türkçe", ("2026-TYT", "SOS"): "TYT Sosyal Bilimler",
    ("2026-TYT", "TEM"): "TYT Temel Matematik", ("2026-TYT", "FEN"): "TYT Fen Bilimleri",
    ("2026-AYT", "TDE-SB1"): "AYT Türk Dili + Sosyal-1", ("2026-AYT", "SB2"): "AYT Sosyal Bilimler-2",
    ("2026-AYT", "MAT"): "AYT Matematik", ("2026-AYT", "FEN"): "AYT Fen Bilimleri",
    ("2026-YDT", "İNG"): "YDT İngilizce", ("2026-YDT", "ALM"): "YDT Almanca",
    ("2026-YDT", "FRA"): "YDT Fransızca", ("2026-YDT", "RUS"): "YDT Rusça",
    ("2026-YDT", "AR"): "YDT Arapça",
}


def report_sections(answers: Dict[str, dict], meta: Dict[str, dict]) -> None:
    """Every printed question, split by the test it belongs to.

    TYT and AYT each have a test called Fen Bilimleri; they are different tests and
    are kept apart here.
    """
    rows: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for i, m in meta.items():
        if i in answers:
            rows[(m["exam"], m["section"])].append(answers[i])
    ordered = sorted(rows.items(),
                     key=lambda kv: -sum(1 for r in kv[1] if r["correct"]) / len(kv[1]))
    print(f"\n{'test':<28}{'accuracy':>10}{'95% CI':>16}{'n':>5}{'figures':>9}")
    print("-" * 68)
    for key, group in ordered:
        hits = sum(1 for r in group if r["correct"])
        lo, hi = wilson(hits, len(group))
        figures = sum(1 for i, m in meta.items()
                      if (m["exam"], m["section"]) == key and m["has_figure"])
        print(f"{NAMES.get(key, key[1]):<28}{100 * hits / len(group):>9.1f}%"
              f"  [{lo:5.1f},{hi:5.1f}]{len(group):>5}{figures:>9}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score a run in YKS nets.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--meta", type=Path, default=Path("data/question_meta.jsonl"))
    parser.add_argument("--variant", default=None, help="restrict to one phrasing, e.g. V5")
    parser.add_argument("--rotation", type=int, default=0)
    args = parser.parse_args(argv)

    meta = {json.loads(l)["id"]: json.loads(l) for l in args.meta.open(encoding="utf-8")}
    answers = load(args.results, args.variant, args.rotation)
    print(f"{len(answers)} of {len(meta)} printed questions answered "
          f"({args.results.name}, variant={args.variant or 'any'}, rotation={args.rotation})")

    report_nets(answers, meta)
    report_sections(answers, meta)
    report_figures(answers, meta)
    report_calibration(answers, meta)
    report_blanking(answers, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
