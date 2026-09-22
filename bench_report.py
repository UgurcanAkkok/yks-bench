#!/usr/bin/env python3
"""Turn the rows from `bench_yks.py` into the benchmark report.

Nothing here calls the model. Every metric is recomputed from the stored
distributions, so the report can change without the numbers being re-earned.

Accuracy carries a Wilson interval everywhere, because at these sample sizes the
interval is usually wider than the gaps people want to read into the table. Two
configurations are only compared with a paired bootstrap over the shared items.

    python bench_report.py data/yks2026/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LABELS = ("A", "B", "C", "D", "E")


# --------------------------------------------------------------------- statistics

def wilson(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (100 * (centre - spread), 100 * (centre + spread))


def accuracy(rows: Sequence[dict]) -> Tuple[float, int, int, Tuple[float, float]]:
    hits = sum(1 for r in rows if r["correct"])
    n = len(rows)
    return (100 * hits / n if n else float("nan"), hits, n, wilson(hits, n))


def paired_bootstrap(a: Dict[str, bool], b: Dict[str, bool], draws: int = 5000,
                     seed: int = 7) -> Tuple[float, Tuple[float, float]]:
    """Difference in accuracy on the items both configurations answered."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return (float("nan"), (float("nan"), float("nan")))
    observed = 100 * (sum(a[k] for k in shared) - sum(b[k] for k in shared)) / len(shared)
    rng = random.Random(seed)
    deltas = []
    for _ in range(draws):
        sample = [shared[rng.randrange(len(shared))] for _ in shared]
        deltas.append(100 * (sum(a[k] for k in sample) - sum(b[k] for k in sample)) / len(sample))
    deltas.sort()
    return observed, (deltas[int(0.025 * draws)], deltas[int(0.975 * draws)])


# -------------------------------------------------------------------- calibration

def top_probability(row: dict) -> float:
    return max(row["probabilities"].values())


def ece(rows: Sequence[dict], bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins."""
    ordered = sorted(rows, key=top_probability)
    if not ordered:
        return float("nan")
    size = max(1, len(ordered) // bins)
    total = 0.0
    for start in range(0, len(ordered), size):
        chunk = ordered[start:start + size]
        if not chunk:
            continue
        mean_p = sum(top_probability(r) for r in chunk) / len(chunk)
        mean_acc = sum(1 for r in chunk if r["correct"]) / len(chunk)
        total += len(chunk) * abs(mean_p - mean_acc)
    return total / len(ordered)


def brier(rows: Sequence[dict]) -> float:
    total = 0.0
    for row in rows:
        for label in LABELS:
            p = row["probabilities"].get(label, 0.0)
            total += (p - (1.0 if label == row["gold"] else 0.0)) ** 2
    return total / len(rows) if rows else float("nan")


def temper(probabilities: Dict[str, float], temperature: float) -> Dict[str, float]:
    """Temperature scaling applied to a distribution: p_T is proportional to p^(1/T)."""
    scaled = {k: max(v, 1e-12) ** (1.0 / temperature) for k, v in probabilities.items()}
    total = sum(scaled.values())
    return {k: v / total for k, v in scaled.items()}


def fit_temperature(rows: Sequence[dict]) -> float:
    """The temperature minimising negative log likelihood of the gold answer."""
    best, best_loss = 1.0, float("inf")
    for step in range(1, 601):
        temperature = 0.05 * step
        loss = 0.0
        for row in rows:
            p = temper(row["probabilities"], temperature).get(row["gold"], 1e-12)
            loss -= math.log(max(p, 1e-12))
        if loss < best_loss:
            best, best_loss = temperature, loss
    return best


def risk_coverage(rows: Sequence[dict]) -> Tuple[Dict[int, float], float]:
    """Accuracy when only the most confident answers are kept."""
    ordered = sorted(rows, key=top_probability, reverse=True)
    points = {}
    for percent in (25, 50, 75, 100):
        kept = ordered[: max(1, len(ordered) * percent // 100)]
        points[percent] = 100 * sum(1 for r in kept if r["correct"]) / len(kept)
    running = []
    hits = 0
    for index, row in enumerate(ordered, start=1):
        hits += row["correct"]
        running.append(hits / index)
    return points, 100 * sum(running) / len(running) if running else float("nan")


# ------------------------------------------------------------------------ loading

def load(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def by_arm(rows: Sequence[dict], arm: str) -> List[dict]:
    return [r for r in rows if r["arm"] == arm]


def correctness(rows: Sequence[dict]) -> Dict[str, bool]:
    return {r["id"]: r["correct"] for r in rows}


def baselines(rows: Sequence[dict]) -> Tuple[float, float]:
    golds = Counter(r["gold"] for r in {r["id"]: r for r in rows}.values())
    n = sum(golds.values())
    return 20.0, (100 * max(golds.values()) / n if n else float("nan"))


def heading(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


# ------------------------------------------------------------------------- report

def report_main(rows: List[dict]) -> None:
    choice = by_arm(rows, "choice")
    random_base, majority = baselines(choice)
    heading("Accuracy by checkpoint and phrasing (rotation 0)")
    print(f"  random {random_base:.1f}%   majority class {majority:.1f}%\n")
    print(f"  {'checkpoint':<17}{'variant':<9}{'acc':>7}{'95% CI':>16}{'n':>6}"
          f"   > random   > majority")
    for checkpoint in sorted({r["checkpoint"] for r in choice}):
        for variant in sorted({r["variant"] for r in choice}):
            subset = [r for r in choice if r["checkpoint"] == checkpoint
                      and r["variant"] == variant and r["rotation"] == 0]
            if not subset:
                continue
            acc, _hits, n, (lo, hi) = accuracy(subset)
            print(f"  {checkpoint:<17}{variant:<9}{acc:6.1f}%  [{lo:5.1f},{hi:5.1f}]{n:6d}"
                  f"   {'yes' if lo > random_base else 'no':<9} {'yes' if lo > majority else 'no'}")
    print("\n  A configuration only counts as better than a baseline when the whole interval")
    print("  clears it. Majority class is the baseline that matters: the gold letters are")
    print("  not uniform, so always answering the commonest one already scores 24.6%.")


def report_rotations(rows: List[dict]) -> None:
    choice = by_arm(rows, "choice")
    rotations = sorted({r["rotation"] for r in choice})
    if len(rotations) < 2:
        return
    heading("Option order: does rotating the five options change the answer?")
    print(f"  {'checkpoint':<17}{'variant':<9}{'mean acc':>10}{'spread':>9}"
          f"{'all agree':>11}{'vote acc':>10}   picked position")
    for checkpoint in sorted({r["checkpoint"] for r in choice}):
        for variant in sorted({r["variant"] for r in choice}):
            per_rotation = []
            picks: Dict[str, List[str]] = defaultdict(list)
            positions = Counter()
            for rotation in rotations:
                subset = [r for r in choice if r["checkpoint"] == checkpoint
                          and r["variant"] == variant and r["rotation"] == rotation]
                if not subset:
                    continue
                per_rotation.append(accuracy(subset)[0])
                for row in subset:
                    picks[row["id"]].append(row["choice"])
                    positions[row["position"]] += 1
            if len(per_rotation) < 2:
                continue
            stable = sum(1 for options in picks.values() if len(set(options)) == 1)
            gold = {r["id"]: r["gold"] for r in choice}
            vote_hits = 0
            for item, options in picks.items():
                if Counter(options).most_common(1)[0][0] == gold[item]:
                    vote_hits += 1
            shape = " ".join(f"{100*positions[p]/sum(positions.values()):4.1f}%" for p in range(5))
            print(f"  {checkpoint:<17}{variant:<9}{sum(per_rotation)/len(per_rotation):9.1f}%"
                  f"{max(per_rotation)-min(per_rotation):8.1f}{100*stable/len(picks):10.1f}%"
                  f"{100*vote_hits/len(picks):9.1f}%   {shape}")
    print("\n  'all agree' = the same option text wins under every rotation.")
    print("  'picked position' = how often the winner sat in slot 1..5 as presented.")


def main_variant(rows: Sequence[dict]) -> str:
    """The phrasing the main arm used; the controls are matched to it."""
    variants = Counter(r["variant"] for r in rows if r["arm"] == "choice")
    return variants.most_common(1)[0][0] if variants else "V1"


def report_controls(rows: List[dict]) -> None:
    heading("Controls: is the benchmark measuring reading, or artifacts?")
    variant = main_variant(rows)
    reference = [r for r in by_arm(rows, "choice")
                 if r["variant"] == variant and r["rotation"] == 0]
    base: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for row in reference:
        base[row["checkpoint"]][row["id"]] = row["correct"]
    print(f"  {'checkpoint':<17}{'arm':<15}{'acc':>7}{'95% CI':>16}{'n':>6}   vs V1 rot0 (paired)")
    for checkpoint in sorted(base):
        for arm in ("choice", "choices-only", "no-passage", "shuffled"):
            subset = [r for r in rows if r["arm"] == arm and r["checkpoint"] == checkpoint
                      and r.get("variant") == variant and r.get("rotation") == 0]
            if not subset:
                continue
            acc, hits, n, (lo, hi) = accuracy(subset)
            delta, (dlo, dhi) = paired_bootstrap(base[checkpoint], correctness(subset))
            tail = "" if arm == "choice" else f"   {delta:+5.1f} [{dlo:+5.1f},{dhi:+5.1f}]"
            print(f"  {checkpoint:<17}{arm:<15}{acc:6.1f}%  [{lo:5.1f},{hi:5.1f}]{n:6d}{tail}")
    print("\n  choices-only sends the five options with an empty state.")
    print("  shuffled pairs each question with another item's state; it must collapse to chance.")


def report_nouls(rows: List[dict]) -> None:
    nouls = by_arm(rows, "noul")
    if not nouls:
        return
    heading("One 5-way Choice vs five independent Nouls")
    print(f"  {'checkpoint':<17}{'variant':<9}{'choice':>9}{'nouls':>9}{'difference (paired)':>24}"
          f"{'mean sum of nouls':>20}")
    for checkpoint in sorted({r["checkpoint"] for r in nouls}):
        for variant in sorted({r["variant"] for r in nouls}):
            noul_rows = [r for r in nouls if r["checkpoint"] == checkpoint and r["variant"] == variant]
            choice_rows = [r for r in by_arm(rows, "choice") if r["checkpoint"] == checkpoint
                           and r["variant"] == variant and r["rotation"] == 0]
            if not noul_rows or not choice_rows:
                continue
            delta, (lo, hi) = paired_bootstrap(correctness(noul_rows), correctness(choice_rows))
            total = sum(r["noul_sum"] for r in noul_rows) / len(noul_rows)
            print(f"  {checkpoint:<17}{variant:<9}{accuracy(choice_rows)[0]:8.1f}%"
                  f"{accuracy(noul_rows)[0]:8.1f}%{delta:+15.1f} [{lo:+5.1f},{hi:+5.1f}]{total:20.3f}")
    print("\n  A Choice is relative and must pick something; each Noul is absolute and may")
    print("  be low for every option. Their sum is not expected to be 1.")


def report_independence(rows: List[dict]) -> None:
    together = {(r["checkpoint"], r["id"]): r for r in by_arm(rows, "together")}
    alone = {(r["checkpoint"], r["id"]): r for r in by_arm(rows, "alone")}
    shared = sorted(set(together) & set(alone))
    if not shared:
        return
    heading("Parallel questions: does asking together change the answer?")
    print(f"  {'checkpoint':<17}{'items':>7}{'argmax flips':>14}{'max |dp|':>11}{'mean |dp|':>12}")
    for checkpoint in sorted({k[0] for k in shared}):
        keys = [k for k in shared if k[0] == checkpoint]
        flips = sum(1 for k in keys if together[k]["choice"] != alone[k]["choice"])
        deltas = [abs(together[k]["probabilities"][label] - alone[k]["probabilities"][label])
                  for k in keys for label in LABELS]
        print(f"  {checkpoint:<17}{len(keys):7d}{flips:>8}/{len(keys):<5}"
              f"{max(deltas):11.4f}{sum(deltas)/len(deltas):12.6f}")
    print("\n  The state is byte-identical in both arms; only the number of questions in")
    print("  the request changes. The documented claim is that this changes nothing.")


def report_calibration(rows: List[dict], seed: int = 11) -> None:
    heading("Calibration and confidence gating")
    variant = main_variant(rows)
    choice = [r for r in by_arm(rows, "choice") if r["rotation"] == 0 and r["variant"] == variant]
    print(f"  {'checkpoint':<17}{'ECE':>7}{'Brier':>8}{'T*':>7}{'ECE after T*':>14}"
          f"{'acc @25%':>10}{'@50%':>8}{'@75%':>8}{'@100%':>8}")
    for checkpoint in sorted({r["checkpoint"] for r in choice}):
        subset = [r for r in choice if r["checkpoint"] == checkpoint]
        if len(subset) < 40:
            continue
        rng = random.Random(seed)
        shuffled = subset[:]
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        fit_half, test_half = shuffled[:half], shuffled[half:]
        temperature = fit_temperature(fit_half)
        rescaled = [dict(r, probabilities=temper(r["probabilities"], temperature)) for r in test_half]
        points, _ = risk_coverage(subset)
        print(f"  {checkpoint:<17}{ece(subset):7.3f}{brier(subset):8.3f}{temperature:7.2f}"
              f"{ece(rescaled):14.3f}{points[25]:9.1f}%{points[50]:7.1f}%"
              f"{points[75]:7.1f}%{points[100]:7.1f}%")
    print("\n  T* is fitted on a random half and the ECE after it is measured on the other half.")
    print("  Coverage columns keep only the most confident answers; a gate is only useful if")
    print("  accuracy climbs as coverage falls.")


def report_sections(rows: List[dict], checkpoint: Optional[str]) -> None:
    variant = main_variant(rows)
    choice = [r for r in by_arm(rows, "choice") if r["rotation"] == 0 and r["variant"] == variant]
    if checkpoint:
        choice = [r for r in choice if r["checkpoint"] == checkpoint]
    if not choice:
        return
    name = checkpoint or "all checkpoints"
    heading(f"By section ({name}, {variant}, rotation 0)")
    print(f"  {'section':<12}{'acc':>7}{'95% CI':>16}{'n':>6}")
    for section in sorted({r["section"] for r in choice}):
        subset = [r for r in choice if r["section"] == section]
        acc, hits, n, (lo, hi) = accuracy(subset)
        print(f"  {section:<12}{acc:6.1f}%  [{lo:5.1f},{hi:5.1f}]{n:6d}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Report on a YKS benchmark run.")
    parser.add_argument("results", type=Path, nargs="?", default=Path("data/yks2026/results.jsonl"))
    parser.add_argument("--sections-for", default=None, help="checkpoint to break down by section")
    args = parser.parse_args(argv)

    rows = load(args.results)
    items = len({r["id"] for r in rows})
    print(f"{len(rows)} rows, {items} distinct questions, "
          f"{len({r['config'] for r in rows})} configurations")

    report_main(rows)
    report_rotations(rows)
    report_controls(rows)
    report_nouls(rows)
    report_independence(rows)
    report_calibration(rows)
    report_sections(rows, args.sections_for)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
