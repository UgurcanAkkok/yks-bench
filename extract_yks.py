#!/usr/bin/env python3
"""Extract questions and answer keys from OSYM YKS booklets into JSONL.

The booklets carry a 45-degree watermark that is drawn glyph by glyph, which is
why naive text extraction interleaves garbage between the option lines.  Every
watermark line has a rotated writing direction, so the body text is recovered by
keeping only the lines whose `dir` is (1, 0).  The rest of this file is layout.

Two facts about the layout do most of the work:

  * A question number lives in its own text block, alone on its line, in a
    narrow band at the left edge of its column.  Numbered items that belong to
    a stimulus ("1. Gun: ...") and the section intro ("1. Bu testte 40 soru
    vardir.") carry their text on the same line, so they never look like a
    question start.
  * A question is contained in one column.  It may run from the right column of
    one page into the left column of the next, but never from left to right.

    python extract_yks.py data/yks2026/*.pdf --out data/yks2026/questions.jsonl

Each record is one question:

    {"id": "2026-TYT/TUR-005", "exam": "2026-TYT", "section": "TUR",
     "section_name": "TURKCE TESTI", "number": 5, "stem": "...",
     "options": {"A": "...", ...}, "gold": "A", "has_figure": false,
     "complete": true, "directions": null, "page": 4,
     "source": "yks_tyt_2026_kitapcik_d350.pdf"}

Every question is emitted, with two flags saying what survived extraction.
`has_figure` means a bitmap sits inside the question, so the text is missing a
table, a graph or a diagram.  `complete` means the stem and all five options
came out as text.  `--text-only` keeps the questions where both are favourable,
which is the subset a text model can actually be scored on.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pymupdf

# Lines whose writing direction is not left-to-right are watermark glyphs.
HORIZONTAL = (1.0, 0.0)

# A question number sits in a narrow band at the left edge of its column; the
# stimulus text of that column starts about 12pt further right.
NUMBER_BAND = 10.0

# Vector art thinner than this in either axis is a rule, not a figure.
FIGURE_MIN = 15.0

# The body of a page, excluding the running header and the footer rule.
BODY_TOP, BODY_BOTTOM = 80.0, 745.0

RE_NUMBER_ONLY = re.compile(r"^(\d{1,3})\.$")
RE_OPTION = re.compile(r"^([A-E])\)\s*(.*)$")
RE_SECTION = re.compile(r"^(20\d\d-(?:TYT|AYT|YDT))/(.+)$")
RE_ANSWER = re.compile(r"^(\d{1,3})\.\s+([A-E]|[İI]PTAL)$", re.IGNORECASE)
# Booklets differ on the punctuation: "16. - 20. sorularda" and "16-20. sorularda".
RE_RANGE = re.compile(r"^(\d{1,3})\.?\s*[-–—]\s*(\d{1,3})\.?\s*sorular", re.IGNORECASE)


def ascii_key(text: str) -> str:
    """Fold a Turkish heading down to something usable in an id."""
    swapped = (text.replace("ı", "i").replace("İ", "I")
                   .replace("ş", "s").replace("Ş", "S")
                   .replace("ğ", "g").replace("Ğ", "G")
                   .replace("ç", "c").replace("Ç", "C")
                   .replace("ö", "o").replace("Ö", "O")
                   .replace("ü", "u").replace("Ü", "U"))
    stripped = unicodedata.normalize("NFKD", swapped).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "-", stripped).strip("-").upper()


@dataclass
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class Row:
    """Everything printed at one height in a column, left to right.

    A question number and the first line of its stem are separate text blocks
    at the same height, and so are an option letter and the word or two that
    follow it, so the row is the unit that actually carries meaning.
    """

    y0: float
    cells: List[Line]

    @property
    def text(self) -> str:
        return " ".join(cell.text for cell in self.cells)


def group_rows(lines: Sequence[Line], tolerance: float = 4.0) -> List[Row]:
    rows: List[Row] = []
    for line in sorted(lines, key=lambda l: (l.y0, l.x0)):
        if rows and line.y0 - rows[-1].y0 <= tolerance:
            rows[-1].cells.append(line)
        else:
            rows.append(Row(line.y0, [line]))
    for row in rows:
        row.cells.sort(key=lambda l: l.x0)
    return rows


@dataclass
class Question:
    exam: str
    section: str
    number: int
    page: int
    stem_lines: List[str] = field(default_factory=list)
    options: Dict[str, List[str]] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    top: float = 0.0
    bottom: float = 0.0
    column: Tuple[float, float] = (0.0, 0.0)
    has_figure: bool = False
    directions: Optional[str] = None
    gold: Optional[str] = None


def page_lines(page: pymupdf.Page) -> List[Line]:
    """Body text of a page, watermark removed, sorted into reading order."""
    out: List[Line] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if tuple(round(v, 2) for v in line["dir"]) != HORIZONTAL:
                continue
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            out.append(Line(x0, y0, x1, y1, text))
    out.sort(key=lambda l: (l.y0, l.x0))
    return out


def figure_rects(page: pymupdf.Page) -> List[pymupdf.Rect]:
    """The bitmaps on a page: tables, graphs, diagrams and geometry.

    Every figure in these booklets is a bitmap.  The vector art is frames and
    rules -- the box around a block of directions, the line above the footer --
    so counting it as a figure flags most of a language test by mistake.
    """
    rects = []
    for info in page.get_image_info():
        rect = pymupdf.Rect(info["bbox"])
        if rect.width < FIGURE_MIN or rect.height < FIGURE_MIN:
            continue
        if rect.y1 < BODY_TOP or rect.y0 > BODY_BOTTOM:
            continue
        rects.append(rect)
    return rects


def section_of(lines: Sequence[Line]) -> Optional[Tuple[str, str]]:
    """The running header, e.g. '2026-TYT/TUR', if this page carries one."""
    for line in lines:
        if line.y0 < 70:
            match = RE_SECTION.match(line.text)
            if match:
                return match.group(1), match.group(2).strip()
    return None


def split_columns(lines: Sequence[Line], mid: float) -> Tuple[List[Line], List[Line]]:
    """Left column, then right column.  A full-width line leads the left."""
    left: List[Line] = []
    right: List[Line] = []
    for line in lines:
        if not BODY_TOP <= line.y0 <= BODY_BOTTOM:   # running header, footer rule
            continue
        (left if line.x0 < mid else right).append(line)
    return left, right


class ColumnParser:
    """Cuts one column of lines into questions, top to bottom."""

    def __init__(self, exam: str, section: str, directions: Dict[Tuple[str, int, int], str]):
        self.exam = exam
        self.section = section
        self.directions = directions
        self.last_number: Optional[int] = None

    def run(
        self,
        lines: Sequence[Line],
        page_no: int,
        bounds: Tuple[float, float],
        carry: Optional[Question],
    ) -> Tuple[List[Question], Optional[Question]]:
        done: List[Question] = []
        current = carry
        if current is not None:
            current.column = bounds
            current.page = page_no
            current.top = BODY_TOP

        left_edge = min((l.x0 for l in lines), default=bounds[0])
        pending: Optional[Tuple[int, int, List[str]]] = None

        def close_pending() -> None:
            nonlocal pending
            if pending is not None:
                low, high, body = pending
                self.directions[(self.section, low, high)] = " ".join(body).strip()
                pending = None

        for row in group_rows(lines):
            text = row.text
            head = row.cells[0]

            span = RE_RANGE.match(text)
            if span:
                close_pending()
                if current is not None:
                    current.bottom = row.y0
                    done.append(current)
                    current = None
                pending = (int(span.group(1)), int(span.group(2)), [text])
                continue

            start = (RE_NUMBER_ONLY.match(head.text)
                     if head.x0 <= left_edge + NUMBER_BAND else None)
            if start and self.accepts(int(start.group(1))):
                close_pending()
                if current is not None:
                    current.bottom = row.y0
                    done.append(current)
                number = int(start.group(1))
                self.last_number = number
                current = Question(exam=self.exam, section=self.section,
                                   number=number, page=page_no)
                current.top = row.y0
                current.column = bounds
                rest = " ".join(cell.text for cell in row.cells[1:]).strip()
                if rest:
                    current.stem_lines.append(rest)
                continue

            if pending is not None:
                pending[2].append(text)
                continue
            if current is None:
                continue

            # Short options are printed two or three to a line, so the option
            # boundary is a cell, not a row.
            for cell in row.cells:
                option = RE_OPTION.match(cell.text)
                if option:
                    label, rest = option.group(1), option.group(2)
                    current.options[label] = [rest] if rest else []
                    current.order.append(label)
                elif current.order:
                    current.options[current.order[-1]].append(cell.text)
                else:
                    current.stem_lines.append(cell.text)

        close_pending()
        if current is not None:
            current.bottom = BODY_BOTTOM
        return done, current

    def accepts(self, number: int) -> bool:
        """Question numbers run 1, 2, 3, ... within a section, with no gaps."""
        if self.last_number is None:
            return number == 1
        return number == self.last_number + 1


def mark_figures(questions: Iterable[Question], rects: Sequence[pymupdf.Rect], page_no: int) -> None:
    for question in questions:
        if question.page != page_no or question.has_figure:
            continue
        x0, x1 = question.column
        region = pymupdf.Rect(x0, question.top, x1, question.bottom)
        if any(region.intersects(rect) for rect in rects):
            question.has_figure = True


def parse_answer_key(page: pymupdf.Page) -> List[Tuple[str, Dict[int, Optional[str]]]]:
    """Read the answer grid as a list of (heading, {number: letter}), left to right."""
    lines = page_lines(page)
    hits = [(l, RE_ANSWER.match(l.text)) for l in lines]
    hits = [(l, m) for l, m in hits if m]
    if len(hits) < 10:
        return []

    columns: List[List[Tuple[Line, re.Match]]] = []
    for line, match in sorted(hits, key=lambda pair: pair[0].x0):
        if columns and line.x0 - columns[-1][-1][0].x0 <= 40:
            columns[-1].append((line, match))
        else:
            columns.append([(line, match)])

    centres = [sum(l.x0 for l, _ in col) / len(col) for col in columns]
    result: List[Tuple[str, Dict[int, Optional[str]]]] = []
    for index, column in enumerate(columns):
        low = 0.0 if index == 0 else (centres[index - 1] + centres[index]) / 2
        high = page.rect.width if index == len(columns) - 1 else (centres[index] + centres[index + 1]) / 2
        first_y = min(line.y0 for line, _ in column)
        heading = " ".join(
            l.text for l in lines
            if 75 < l.y0 < first_y and low <= l.x0 < high and not RE_ANSWER.match(l.text)
        )
        answers: Dict[int, Optional[str]] = {}
        for line, match in column:
            letter = match.group(2).upper()
            answers[int(match.group(1))] = letter if letter in "ABCDE" else None
        result.append((re.sub(r"\s+", " ", heading).strip(), answers))

    merged: List[Tuple[str, Dict[int, Optional[str]]]] = []
    for heading, answers in result:
        if merged and merged[-1][0] == heading:
            merged[-1][1].update(answers)
        else:
            merged.append((heading, answers))
    return merged


def extract(path: Path) -> Tuple[List[Question], List[Tuple[str, Dict[int, Optional[str]]]]]:
    doc = pymupdf.open(path)
    questions: List[Question] = []
    key: List[Tuple[str, Dict[int, Optional[str]]]] = []
    directions: Dict[Tuple[str, int, int], str] = {}
    parser: Optional[ColumnParser] = None
    header: Optional[Tuple[str, str]] = None
    carry: Optional[Question] = None

    for index, page in enumerate(doc):
        page_no = index + 1
        lines = page_lines(page)
        found = section_of(lines)

        if found is None:
            grid = parse_answer_key(page)
            if grid:
                key = grid
            if carry is not None:                  # a cover or the key ends the run
                questions.append(carry)
                carry = None
            continue

        if found != header:
            if carry is not None:
                questions.append(carry)
                carry = None
            header = found
            parser = ColumnParser(found[0], found[1], directions)

        assert parser is not None
        mid = page.rect.width / 2
        columns = split_columns(lines, mid)
        fresh: List[Question] = []
        for position, column_lines in enumerate(columns):
            bounds = (0.0, mid) if position == 0 else (mid, page.rect.width)
            batch, carry = parser.run(column_lines, page_no, bounds, carry)
            fresh.extend(batch)
            if position == 0 and carry is not None:
                fresh.append(carry)                # no question spans the gutter
                carry = None

        pending = fresh + ([carry] if carry is not None else [])
        mark_figures(pending, figure_rects(page), page_no)
        questions.extend(fresh)

    if carry is not None:
        questions.append(carry)

    for question in questions:
        for (section, low, high), text in directions.items():
            if section == question.section and low <= question.number <= high:
                question.directions = text
    return questions, key


def attach_gold(
    questions: Sequence[Question], key: Sequence[Tuple[str, Dict[int, Optional[str]]]]
) -> Dict[str, str]:
    """Map key columns onto sections by order of first appearance in the booklet."""
    order: List[str] = []
    for question in questions:
        if question.section not in order:
            order.append(question.section)

    names: Dict[str, str] = {}
    gold: Dict[Tuple[str, int], Optional[str]] = {}
    for position, (heading, answers) in enumerate(key):
        if position >= len(order):
            break
        section = order[position]
        names[section] = heading
        for number, letter in answers.items():
            gold[(section, number)] = letter

    for question in questions:
        question.gold = gold.get((question.section, question.number))
    return names


def clean(parts: Sequence[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def is_complete(stem: str, options: Dict[str, str], directions: Optional[str]) -> bool:
    """Whether the text alone is enough to answer.

    A blank option means the choice was drawn rather than typed -- a fraction or
    a formula -- which a bitmap check does not always catch.  A blank stem is
    fine when the question is a numbered gap in a shared passage, because the
    passage is in `directions`.
    """
    if len(options) != 5 or not all(value.strip() for value in options.values()):
        return False
    return bool(stem.strip() or directions)


def record(question: Question, names: Dict[str, str], source: str) -> dict:
    stem = clean(question.stem_lines)
    options = {label: clean(question.options[label]) for label in sorted(question.options)}
    return {
        "id": f"{question.exam}/{ascii_key(question.section)}-{question.number:03d}",
        "exam": question.exam,
        "section": question.section,
        "section_name": names.get(question.section, ""),
        "number": question.number,
        "stem": stem,
        "options": options,
        "gold": question.gold,
        "has_figure": question.has_figure,
        "complete": is_complete(stem, options, question.directions),
        "directions": question.directions,
        "page": question.page,
        "source": source,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract YKS questions and answer keys into JSONL.")
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="JSONL destination; stdout when omitted")
    parser.add_argument("--text-only", action="store_true",
                        help="keep only questions answerable from text alone")
    parser.add_argument("--quiet", action="store_true", help="suppress the per-section report")
    args = parser.parse_args(argv)

    rows: List[dict] = []
    report: List[str] = []
    for path in args.pdfs:
        questions, key = extract(path)
        names = attach_gold(questions, key)
        buckets: Dict[str, List[Question]] = defaultdict(list)
        for question in questions:
            buckets[question.section].append(question)

        report.append(path.name)
        for section, group in buckets.items():
            numbers = [q.number for q in group]
            missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers)) if numbers else []
            duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
            complete = sum(1 for q in group if len(q.options) == 5)
            with_gold = sum(1 for q in group if q.gold)
            figures = sum(1 for q in group if q.has_figure)
            usable = sum(
                1 for q in group
                if not q.has_figure
                and is_complete(clean(q.stem_lines),
                                {k: clean(v) for k, v in q.options.items()}, q.directions)
            )
            report.append(
                f"  {section:<10} {len(group):>3} questions  {complete:>3} with 5 options  "
                f"{with_gold:>3} with gold  {figures:>3} with figures  "
                f"{usable:>3} text-only  -> {names.get(section, '?')}"
            )
            if missing:
                report.append(f"             missing numbers: {missing}")
            if duplicates:
                report.append(f"             duplicate numbers: {duplicates}")

        for question in questions:
            row = record(question, names, path.name)
            if args.text_only and (row["has_figure"] or not row["complete"]):
                continue
            rows.append(row)

    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if args.out:
        args.out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if not args.quiet:
        print("\n".join(report), file=sys.stderr)
        print(f"\n{len(rows)} questions written" + (f" to {args.out}" if args.out else ""),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
