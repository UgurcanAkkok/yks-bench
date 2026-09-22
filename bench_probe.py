#!/usr/bin/env python3
"""Diagnostics for a benchmark result that came out at chance.

An at-chance score has two very different explanations: the model is broken or
mis-configured, or the task is outside what it can do. These probes separate them.

  sanity     tasks the model is built for, with an obvious right answer. If these
             fail, nothing else in the benchmark means anything.
  ladder     cloze questions in increasing order of subtlety, to find the rung
             where discrimination stops working.
  options    the exam questions and states unchanged, with the four wrong options
             replaced by progressively easier ones. Isolates reading from
             discrimination: if the model can find the gold answer among obvious
             non-answers, it is reading and failing to discriminate.
  state      the exam options unchanged, with the state cut down or removed.
             Isolates how much the passage contributes.

    python bench_probe.py sanity ladder options state
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_yks import Client, question_sentence, state_for   # noqa: E402

LABELS = ("A", "B", "C", "D", "E")
GENERIC = "Aşağıdaki seçeneklerden hangisi bu sorunun doğru cevabıdır?"

# Short concrete nouns. No reader would offer one as the answer to an exam question.
CONCRETE = ["elma", "otomobil", "mavi renk", "deniz kenarı", "tahta masa", "kırmızı bisiklet",
            "sıcak çay", "eski ayakkabı", "camdan bardak", "yeşil çimen", "demir kapı", "beyaz bulut"]


def wilson(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def choice(instructions: str, criteria: Dict[str, str]) -> Dict:
    return {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}}


# ------------------------------------------------------------------------- sanity

SANITY = [
    ("EN route: refund", "english",
     {"message": "I was charged twice for order A-104. Please refund the duplicate."},
     "What does the customer want in `message`?",
     {"refund": "money returned or a duplicate charge reversed",
      "technical_help": "a bug, outage or integration problem",
      "sales": "pricing, plans or buying", "other": "anything else"}, "refund"),
    ("EN route: bug", "english",
     {"message": "Your API returns 500 on every POST to /v1/orders since this morning."},
     "What does the customer want in `message`?",
     {"refund": "money returned or a duplicate charge reversed",
      "technical_help": "a bug, outage or integration problem",
      "sales": "pricing, plans or buying", "other": "anything else"}, "technical_help"),
    ("EN sentiment", "english",
     {"review": "Absolutely terrible. It broke on day one and support ignored me."},
     "Is the `review` positive or negative?",
     {"positive": "the writer is pleased", "negative": "the writer is unhappy"}, "negative"),
    ("EN world knowledge", "english", "The capital city of France is being discussed.",
     "Which city is the capital of France?",
     {"Paris": "Paris", "Berlin": "Berlin", "Madrid": "Madrid",
      "Rome": "Rome", "Lisbon": "Lisbon"}, "Paris"),
    ("EN category", "english", "A cat is a small domesticated mammal that meows.",
     "What kind of thing is a cat?",
     {"animal": "a living creature", "plant": "a living plant",
      "vehicle": "a machine for transport", "building": "a structure",
      "colour": "a colour"}, "animal"),
    ("TR route: refund", "multilingual",
     {"mesaj": "A-104 numaralı siparişim için iki kez ücret alındı. Lütfen iade edin."},
     "`mesaj` içinde müşteri ne istiyor?",
     {"iade": "paranın geri verilmesi", "teknik_destek": "bir hata veya kesinti",
      "satis": "fiyat veya satın alma", "diger": "başka bir şey"}, "iade"),
    ("TR sentiment", "multilingual",
     {"yorum": "Berbat bir ürün. İlk gün bozuldu ve destek hiç ilgilenmedi."},
     "`yorum` olumlu mu olumsuz mu?",
     {"olumlu": "yazar memnun", "olumsuz": "yazar memnun değil"}, "olumsuz"),
    ("TR world knowledge", "multilingual", "Türkiye'nin başkenti konuşuluyor.",
     "Türkiye'nin başkenti hangisidir?",
     {"Ankara": "Ankara", "İstanbul": "İstanbul", "İzmir": "İzmir",
      "Bursa": "Bursa", "Antalya": "Antalya"}, "Ankara"),
    ("TR category", "multilingual", "Kedi miyavlayan küçük bir evcil memelidir.",
     "Kedi ne tür bir şeydir?",
     {"hayvan": "canlı bir yaratık", "bitki": "canlı bir bitki",
      "araç": "ulaşım makinesi", "bina": "yapı", "renk": "bir renk"}, "hayvan"),
    ("TR reading", "multilingual", "Ali pazardan üç kilo elma aldı. Ayşe ise armut aldı.",
     "Ali pazardan ne aldı?",
     {"elma": "elma", "armut": "armut", "muz": "muz", "üzüm": "üzüm", "kiraz": "kiraz"}, "elma"),
]

INS_EN = "Which word best fits the blank marked ---- in `sentence`?"
INS_TR = "`cümle` içindeki ---- yerine hangi sözcük gelmelidir?"

LADDER = [
    ("EN absurd distractors", "english", INS_EN,
     {"sentence": "The cat ---- on the warm mat and fell asleep."},
     {w: w for w in ("sat", "dissolved", "computed", "photosynthesised", "invoiced")}, "sat"),
    ("EN same part of speech", "english", INS_EN,
     {"sentence": "She ---- the door quietly so nobody would wake up."},
     {w: w for w in ("closed", "shouted", "swam", "melted", "argued")}, "closed"),
    ("EN collocation", "english", INS_EN,
     {"sentence": "The committee decided to ---- a decision until next month."},
     {w: w for w in ("postpone", "prolong", "suspend", "withdraw", "dismiss")}, "postpone"),
    ("EN discourse connective", "english", INS_EN,
     {"sentence": "Solar panels are cheap to run. ----, the up-front cost is high."},
     {w: w for w in ("However", "Therefore", "Similarly", "Moreover", "Meanwhile")}, "However"),
    ("TR absurd distractors", "multilingual", INS_TR,
     {"cümle": "Kedi sıcak minderin üzerine ---- ve uyuyakaldı."},
     {w: w for w in ("oturdu", "çözündü", "faturalandı", "fotosentez yaptı", "derlendi")}, "oturdu"),
    ("TR same part of speech", "multilingual", INS_TR,
     {"cümle": "Kimse uyanmasın diye kapıyı sessizce ----."},
     {w: w for w in ("kapattı", "bağırdı", "yüzdü", "eridi", "tartıştı")}, "kapattı"),
    ("TR discourse connective", "multilingual", INS_TR,
     {"cümle": "Güneş panelleri ucuza çalışır. ----, ilk kurulum maliyeti yüksektir."},
     {w: w for w in ("Ancak", "Bu yüzden", "Benzer şekilde", "Ayrıca", "Bu arada")}, "Ancak"),
]


def run_fixed(client: Client, cases: Sequence, title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    print(f"  {'case':<26}{'checkpoint':<15}{'picked':<20}{'p':>7}{'conf':>8}   verdict")
    hits = 0
    for label, checkpoint, state, instructions, criteria, want in cases:
        payload, _ = client.ask(state, choice(instructions, criteria), checkpoint)
        answer = payload["answers"]["q"]
        picked = answer["choice"]
        hits += picked == want
        confidence = answer.get("confidence")
        print(f"  {label:<26}{checkpoint:<15}{picked:<20}"
              f"{answer['probabilities'][picked]:7.3f}"
              f"{(confidence if confidence is not None else float('nan')):8.3f}   "
              f"{'OK' if picked == want else 'WRONG (want ' + want + ')'}")
    print(f"\n  {hits}/{len(cases)} correct")


# -------------------------------------------------------------------- exam probes

def load_rows(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def borrowed(rows: List[dict], row: dict, rng: random.Random, same_section: bool) -> List[str]:
    pool = [r for r in rows if (r["section"] == row["section"]) == same_section and r["id"] != row["id"]]
    picks, seen = [], {row["options"][row["gold"]]}
    while len(picks) < 4:
        donor = pool[rng.randrange(len(pool))]
        text = donor["options"][rng.choice(LABELS)]
        if text.strip() and text not in seen:
            seen.add(text)
            picks.append(text)
    return picks


def sweep(client: Client, rows: List[dict], checkpoint: str, title: str,
          arms: Sequence[Tuple[str, Callable[[dict, random.Random], Tuple[Dict[str, str], str]],
                              Callable[[dict], object]]]) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    print(f"  {'arm':<38}{'acc':>7}{'95% CI':>16}")
    for label, make_options, make_state in arms:
        rng = random.Random(20260620)
        hits = 0
        for row in rows:
            criteria, gold = make_options(row, rng)
            payload, _ = client.ask(make_state(row), choice(GENERIC, criteria), checkpoint)
            hits += payload["answers"]["q"]["choice"] == gold
        lo, hi = wilson(hits, len(rows))
        print(f"  {label:<38}{100*hits/len(rows):6.1f}%  [{lo:5.1f},{hi:5.1f}]")


def place(gold_text: str, distractors: List[str], rng: random.Random) -> Tuple[Dict[str, str], str]:
    slot = rng.randrange(5)
    texts = list(distractors)
    texts.insert(slot, gold_text)
    return {label: texts[i] for i, label in enumerate(LABELS)}, LABELS[slot]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose an at-chance benchmark result.")
    parser.add_argument("probes", nargs="*", default=["sanity", "ladder", "options", "state"],
                        choices=["sanity", "ladder", "options", "state"])
    parser.add_argument("--questions", type=Path, default=Path("data/yks2026/questions_textonly.jsonl"))
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--checkpoint", default="multilingual")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    client = Client(args.url)
    if "sanity" in args.probes:
        run_fixed(client, SANITY, "Sanity: tasks the model is built for")
    if "ladder" in args.probes:
        run_fixed(client, LADDER, "Ladder: cloze by increasing subtlety")

    if "options" in args.probes or "state" in args.probes:
        rows = load_rows(args.questions)[: args.limit]
        real = lambda row, rng: ({l: row["options"][l] for l in LABELS}, row["gold"])
        full = lambda row: state_for(row, "V1")

        if "options" in args.probes:
            sweep(client, rows, args.checkpoint,
                  f"Options: same states, easier wrong answers ({args.checkpoint})", [
                      ("real options (as ÖSYM wrote them)", real, full),
                      ("borrowed from the same section",
                       lambda row, rng: place(row["options"][row["gold"]],
                                              borrowed(rows, row, rng, True), rng), full),
                      ("borrowed from another section",
                       lambda row, rng: place(row["options"][row["gold"]],
                                              borrowed(rows, row, rng, False), rng), full),
                      ("four concrete nouns",
                       lambda row, rng: place(row["options"][row["gold"]],
                                              rng.sample(CONCRETE, 4), rng), full),
                  ])

        if "state" in args.probes:
            sweep(client, rows, args.checkpoint,
                  f"State: real options, less context ({args.checkpoint})", [
                      ("full state", real, full),
                      ("question sentence only", real,
                       lambda row: question_sentence(row["stem"])[1] or row["stem"][-160:]),
                      ("no state at all", real, lambda row: ""),
                  ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
