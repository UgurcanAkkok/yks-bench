#!/usr/bin/env python3
"""Run Laya against the 2026 YKS exam and record every answer.

This half only collects; `bench_report.py` computes the metrics. The split is
deliberate: the full probability distribution for every item is written to disk,
so any metric can be recomputed later without paying for inference again.

What gets run, and why each one is here:

  variants     V1/V2/V3 phrase the same question three ways. This is not a search
               for the best prompt -- all three are reported. For a model whose
               promise is "write a question, get a calibrated answer", how much
               the answer moves with the phrasing is itself the result.
  rotations    The five options are cyclically rotated. Laya has no answer-letter
               tokens, so the LLM mechanism for position bias cannot apply here;
               but each option is scored at a [MASK] in a different position with
               different neighbours, which is its own reason to check.
  nouls        The same item as one 5-way Choice and as five independent Nouls.
               The docs say these answer different questions; gold labels settle
               which decomposition is better.
  choices-only The options with an empty state. If accuracy survives that, the
               benchmark is measuring artifacts in the option set, not reading.
  no-passage   Passage-based items with the passage removed.
  shuffled     Each question paired with another item's state. Must collapse to
               chance, or the harness is leaking gold somewhere.

Run it unbatched. Concurrent micro-batching shifts probabilities by up to 0.04
in bf16, which is wider than the top-1 margin on a quarter of these items.

    python bench_yks.py --out data/yks2026/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

LABELS = ("A", "B", "C", "D", "E")

GENERIC_INSTRUCTION = "Aşağıdaki seçeneklerden hangisi bu sorunun doğru cevabıdır?"
CLOZE_INSTRUCTION = "Cümlede ---- ile gösterilen boşluğa aşağıdakilerden hangisi gelmelidir?"
FIELD_INSTRUCTION = "`soru` sorusunun doğru cevabı aşağıdakilerden hangisidir?"
NOUL_INSTRUCTION = "`soru` sorusunun doğru cevabı şu seçenek midir: %s"


def question_sentence(stem: str) -> Tuple[str, str]:
    """Split a YKS stem into its passage and its closing interrogative sentence."""
    asked = re.findall(r"[^.?!]*\?", stem)
    if not asked:
        return stem, ""
    tail = asked[-1].strip()
    return stem[: stem.rfind(tail)].strip(), tail


def state_for(row: dict, variant: str, drop_passage: bool = False) -> Any:
    """The state for one item.

    The question always comes first. Laya truncates a state from the right, so if
    a budget were ever exceeded the passage would be cut before the question is.
    """
    directions = "" if drop_passage else (row["directions"] or "")
    if variant == "V1":
        return " ".join(part for part in (row["stem"], directions) if part)
    if variant == "V2":
        return {k: v for k, v in (("soru", row["stem"]), ("parça", directions)) if v}
    if variant == "V3":
        passage, asked = question_sentence(row["stem"])
        context = " ".join(part for part in (passage, directions) if part)
        return {k: v for k, v in (("soru", asked or row["stem"]), ("parça", context)) if v}
    if variant in ("V4", "V5"):
        # The form the docs prescribe: the state carries the content and nothing
        # else, and the judgment being asked for lives in `instructions`. For a
        # question that ends in an interrogative, that sentence moves out of the
        # state entirely; for a cloze item the sentence with the blank is content.
        passage, asked = question_sentence(row["stem"])
        content = (passage if asked else row["stem"])
        return " ".join(part for part in (content, directions) if part)
    raise ValueError(f"unknown variant {variant!r}")


def instruction_for(row: dict, variant: str) -> str:
    if variant == "V1":
        return GENERIC_INSTRUCTION
    if variant == "V2":
        return FIELD_INSTRUCTION
    asked = question_sentence(row["stem"])[1]
    if variant in ("V4", "V5"):
        return asked or CLOZE_INSTRUCTION
    return asked or row["stem"]


def rotate(labels: Sequence[str], by: int) -> List[str]:
    by %= len(labels)
    return list(labels[by:]) + list(labels[:by])


def choice_question(row: dict, variant: str, rotation: int, encoding: str = "bare"
                    ) -> Tuple[Dict[str, Any], List[str], Dict[str, str]]:
    """One 5-way Choice. The criteria dict's order is the order the model sees.

    laya renders a choice option as "<key>: <description>", so criteria keyed by
    answer letter put a bare "A: ", "B: " in front of every option -- a token that
    carries no information about the answer and measurably moves the result.
    Keying by the option text with an empty description renders the option alone,
    which is what `bare` does and what this benchmark uses.
    """
    order = rotate(LABELS, rotation)
    if encoding == "lettered":
        criteria = {label: row["options"][label] for label in order}
        back = {label: label for label in order}
    else:
        criteria = {row["options"][label]: "" for label in order}
        back = {row["options"][label]: label for label in order}
    question = {"type": "choice", "instructions": instruction_for(row, variant), "criteria": criteria}
    return {"q": question}, order, back


def noul_questions(row: dict, variant: str) -> Dict[str, Any]:
    """The same item as five independent yes/no questions, one per option."""
    return {
        label: {"type": "noul",
                "instructions": NOUL_INSTRUCTION % row["options"][label] if variant != "V1"
                                else f"Bu sorunun doğru cevabı şu seçenek midir: {row['options'][label]}",
                "criteria": {"true": "evet, doğru cevap budur",
                             "false": "hayır, doğru cevap bu değildir"}}
        for label in LABELS
    }


class Client:
    def __init__(self, url: str, timeout: float = 120.0, api_key: Optional[str] = None):
        self.url = url.rstrip("/") + "/v1/systemone"
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.calls = 0
        self.input_tokens = 0

    def ask(self, state: Any, questions: Dict[str, Any], checkpoint: str) -> Tuple[dict, float]:
        body = json.dumps({"state": state, "model": checkpoint, "questions": questions}).encode()
        request = urllib.request.Request(self.url, data=body, headers=self.headers)
        started = time.perf_counter()
        payload = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                    payload = json.load(handle)
                break
            except urllib.error.HTTPError as exc:
                # 503: the local server's queue is full. 429: the hosted API's rate
                # limit, which carries a retry-after the SDKs are documented to honour.
                if exc.code in (429, 503) and attempt < 5:
                    wait = exc.headers.get("retry-after") if exc.headers else None
                    try:
                        delay = float(wait) if wait else 0.0
                    except ValueError:
                        delay = 0.0
                    time.sleep(max(delay, 0.5 * (attempt + 1)))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < 5:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
        if payload is None:
            raise RuntimeError("no response after retries")
        self.calls += 1
        self.input_tokens += int(payload.get("usage", {}).get("input_tokens", 0) or 0)
        return payload, (time.perf_counter() - started) * 1000


def base_record(row: dict, arm: str, checkpoint: str, variant: str, rotation: int) -> dict:
    return {
        "arm": arm,
        "config": f"{checkpoint}/{variant}/rot{rotation}" if arm == "choice" else f"{checkpoint}/{variant}/{arm}",
        "checkpoint": checkpoint,
        "variant": variant,
        "rotation": rotation,
        "id": row["id"],
        "exam": row["exam"],
        "section": row["section"],
        "gold": row["gold"],
    }


def run_choice(client: Client, rows: Sequence[dict], checkpoint: str, variant: str,
               rotation: int, arm: str = "choice", states: Optional[Dict[str, Any]] = None,
               encoding: str = "bare", workers: int = 1) -> Iterator[dict]:
    if workers > 1:
        # Only for a remote endpoint, where our own batching cannot perturb anything.
        # Against the local server this must stay at 1: concurrent micro-batching
        # changes padding and moves probabilities.
        def one(row):
            return next(iter(run_choice(client, [row], checkpoint, variant, rotation,
                                        arm, states, encoding, workers=1)))
        with ThreadPoolExecutor(workers) as pool:
            yield from pool.map(one, rows)
        return
    for row in rows:
        questions, order, back = choice_question(row, variant, rotation, encoding)
        state = states[row["id"]] if states is not None else state_for(row, variant)
        payload, latency = client.ask(state, questions, checkpoint)
        answer = payload["answers"]["q"]
        picked = back[answer["choice"]]
        probabilities = {back[k]: v for k, v in answer["probabilities"].items()}
        record = base_record(row, arm, checkpoint, variant, rotation)
        record.update({
            "choice": picked,
            "probabilities": probabilities,
            "confidence": answer["confidence"],
            "correct": picked == row["gold"],
            "presented": order,
            "position": order.index(picked),
            "encoding": encoding,
            "served_by": payload.get("model"),
            "latency_ms": round(latency, 2),
        })
        yield record


def run_nouls(client: Client, rows: Sequence[dict], checkpoint: str, variant: str,
              workers: int = 1) -> Iterator[dict]:
    """Five Nouls in one request: one state, five questions, one forward pass."""
    if workers > 1:
        def one(row):
            return next(iter(run_nouls(client, [row], checkpoint, variant, workers=1)))
        with ThreadPoolExecutor(workers) as pool:
            yield from pool.map(one, rows)
        return
    for row in rows:
        state = state_for(row, variant)
        payload, latency = client.ask(state, noul_questions(row, variant), checkpoint)
        answers = payload["answers"]
        scores = {label: answers[label]["noul"] for label in LABELS}
        pick = max(scores, key=lambda label: scores[label])
        record = base_record(row, "noul", checkpoint, variant, 0)
        record.update({
            "choice": pick,
            "probabilities": scores,
            "confidence": None,
            "correct": pick == row["gold"],
            "noul_sum": round(sum(scores.values()), 4),
            "served_by": payload.get("model"),
            "latency_ms": round(latency, 2),
        })
        yield record


def run_independence(client: Client, rows: Sequence[dict], checkpoint: str, variant: str) -> Iterator[dict]:
    """Shared-passage items asked together vs one at a time.

    The docs say questions in one request are evaluated independently. These groups
    are the shape that claim covers: one state, several questions.
    """
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        if row["directions"]:
            groups.setdefault((row["section"], row["directions"]), []).append(row)
    for (_section, directions), members in groups.items():
        if len(members) < 2:
            continue
        state = {"parça": directions,
                 "sorular": {row["id"]: row["stem"] for row in members}}
        questions = {}
        for row in members:
            questions[row["id"]] = {
                "type": "choice",
                "instructions": f"`sorular.{row['id']}` sorusunun doğru cevabı hangisidir?",
                "criteria": {label: row["options"][label] for label in LABELS},
            }
        payload, latency = client.ask(state, questions, checkpoint)
        for row in members:
            answer = payload["answers"][row["id"]]
            record = base_record(row, "together", checkpoint, variant, 0)
            record.update({
                "choice": answer["choice"],
                "probabilities": answer["probabilities"],
                "confidence": answer["confidence"],
                "correct": answer["choice"] == row["gold"],
                "group_size": len(members),
                "latency_ms": round(latency, 2),
            })
            yield record

        # The same question on its own, against a byte-identical state. Only the
        # number of questions in the request changes, which is the claim being tested.
        for row in members:
            solo_q = {row["id"]: questions[row["id"]]}
            payload, latency = client.ask(state, solo_q, checkpoint)
            answer = payload["answers"][row["id"]]
            record = base_record(row, "alone", checkpoint, variant, 0)
            record.update({
                "choice": answer["choice"],
                "probabilities": answer["probabilities"],
                "confidence": answer["confidence"],
                "correct": answer["choice"] == row["gold"],
                "group_size": len(members),
                "latency_ms": round(latency, 2),
            })
            yield record


def read_env_key(path: Path = Path(".env")) -> Optional[str]:
    """TYPESAFE_API_KEY from a local .env, so the key never reaches a command line."""
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "TYPESAFE_API_KEY":
            return value.strip().strip("'\"") or None
    return None


def load(path: Path, limit: Optional[int], exams: Optional[Sequence[str]]) -> List[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if exams:
        rows = [r for r in rows if r["exam"] in exams]
    return rows[:limit] if limit else rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Laya against the 2026 YKS exam.")
    parser.add_argument("--questions", type=Path, default=Path("data/yks2026/questions_textonly.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/yks2026/results.jsonl"))
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--api-key", default=None,
                        help="bearer token; also read from TYPESAFE_API_KEY or ./.env")
    parser.add_argument("--checkpoints", default="multilingual,english,typed-decisions")
    parser.add_argument("--variants", default="V1,V2,V3")
    parser.add_argument("--rotations", type=int, default=5, help="cyclic option rotations per config")
    parser.add_argument("--option-encoding", choices=("bare", "lettered"), default="bare",
                        help="'lettered' puts 'A: ' in front of every option, as laya renders "
                             "letter-keyed criteria; 'bare' sends the option text alone")
    parser.add_argument("--exams", default=None, help="restrict to e.g. 2026-TYT,2026-AYT")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="concurrent requests; keep at 1 for the local server")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--resume", action="store_true", help="keep rows already in --out")
    args = parser.parse_args(argv)

    api_key = args.api_key or os.environ.get("TYPESAFE_API_KEY") or read_env_key()
    rows = load(args.questions, args.limit, args.exams.split(",") if args.exams else None)
    checkpoints = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    client = Client(args.url, api_key=api_key)

    done: set = set()
    mode = "a"
    if args.resume and args.out.exists():
        for line in args.out.open(encoding="utf-8"):
            record = json.loads(line)
            done.add((record["config"], record["id"], record["arm"]))
        print(f"resuming: {len(done)} rows already recorded", file=sys.stderr)
    else:
        mode = "w"

    def remaining(candidates: Sequence[dict], checkpoint: str, variant: str,
                  arm: str, rotation: int = 0) -> List[dict]:
        """Rows this config still needs.

        Filtering here rather than at the sink is what makes --resume cheap: a row
        that is already recorded costs nothing instead of costing a request whose
        answer is then thrown away.
        """
        config = (f"{checkpoint}/{variant}/rot{rotation}" if arm == "choice"
                  else f"{checkpoint}/{variant}/{arm}")
        return [r for r in candidates if (config, r["id"], arm) not in done]

    def unfinished_groups(candidates: Sequence[dict], checkpoint: str, variant: str) -> List[dict]:
        """Members of shared-passage groups that are not fully recorded in both arms."""
        groups: Dict[Tuple[str, str], List[dict]] = {}
        for row in candidates:
            if row["directions"]:
                groups.setdefault((row["section"], row["directions"]), []).append(row)
        out: List[dict] = []
        for members in groups.values():
            if len(members) < 2:
                continue
            complete = all(
                (f"{checkpoint}/{variant}/{arm}", row["id"], arm) in done
                for row in members for arm in ("together", "alone"))
            if not complete:
                out.extend(members)
        return out

    started = time.perf_counter()
    written = 0
    with args.out.open(mode, encoding="utf-8") as sink:
        def emit(stream: Iterator[dict]) -> None:
            nonlocal written
            for record in stream:
                if (record["config"], record["id"], record["arm"]) in done:
                    continue
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written % 500 == 0:
                    sink.flush()
                    rate = written / (time.perf_counter() - started)
                    print(f"  {written} rows  ({rate:.0f}/s)", file=sys.stderr)

        for checkpoint in checkpoints:
            for variant in variants:
                for rotation in range(args.rotations):
                    print(f"choice  {checkpoint}/{variant}/rot{rotation}", file=sys.stderr)
                    emit(run_choice(client, remaining(rows, checkpoint, variant, "choice", rotation),
                                     checkpoint, variant, rotation,
                                     encoding=args.option_encoding, workers=args.workers))

                print(f"nouls   {checkpoint}/{variant}", file=sys.stderr)
                emit(run_nouls(client, remaining(rows, checkpoint, variant, "noul"),
                               checkpoint, variant, workers=args.workers))

            if args.skip_controls:
                continue

            # The controls must use the same phrasing as the main arm, or a paired
            # difference would mix "the state changed" with "the question changed".
            control = variants[0]

            print(f"control {checkpoint}/choices-only", file=sys.stderr)
            empty = {row["id"]: "" for row in rows}
            emit(run_choice(client, remaining(rows, checkpoint, control, "choices-only"),
                             checkpoint, control, 0, arm="choices-only",
                             states=empty, encoding=args.option_encoding, workers=args.workers))

            passage_rows = [row for row in rows if row["directions"]]
            print(f"control {checkpoint}/no-passage ({len(passage_rows)} items)", file=sys.stderr)
            stripped = {row["id"]: state_for(row, control, drop_passage=True) for row in passage_rows}
            emit(run_choice(client, remaining(passage_rows, checkpoint, control, "no-passage"),
                             checkpoint, control, 0, arm="no-passage",
                             states=stripped, encoding=args.option_encoding, workers=args.workers))

            print(f"control {checkpoint}/shuffled", file=sys.stderr)
            rng = random.Random(args.seed)
            donors = list(rows)
            rng.shuffle(donors)
            mismatched = {}
            for row, donor in zip(rows, donors):
                if donor["id"] == row["id"]:
                    donor = donors[(donors.index(donor) + 1) % len(donors)]
                mismatched[row["id"]] = state_for(donor, control)
            emit(run_choice(client, remaining(rows, checkpoint, control, "shuffled"),
                             checkpoint, control, 0, arm="shuffled",
                             states=mismatched, encoding=args.option_encoding, workers=args.workers))

            print(f"parallel {checkpoint}/independence", file=sys.stderr)
            emit(run_independence(client, unfinished_groups(rows, checkpoint, control), checkpoint, control))

    elapsed = time.perf_counter() - started
    print(f"\n{written} rows written to {args.out} in {elapsed:.0f}s "
          f"({client.calls} requests, {1000*elapsed/max(1,client.calls):.1f} ms/request, "
          f"{client.input_tokens} input tokens)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
