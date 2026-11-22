"""Prove by text overlap that the held-out questions have not leaked into anything that shaped the system.

The 30 held-out questions were committed before the system was run on them, and nobody who changes the system
may read them before the freeze. Had a held-out question or its gold answer reached a prompt, an example pool,
a training file, a learned store, a cached model call, an oracle plan or a result, its words would be there.
This script looks for every held-out question, and separately for its gold answer, in every text file of the
repo and in the development material outside it, in three ways:

    presence     the whole text appears in the file;
    containment  the share of its word 6-grams found anywhere in the file (flagged at 0.5);
    line         the best difflib ratio against one short line (under 400 characters) in the files where a
                 reworded question would sit: code, training data, learned stores and the development question
                 sets (flagged at 0.85). A JSON string counts as a line of its own.

Text is normalised first: string escapes undone, lower case, letters and digits only, single spaces, so quotes,
escapes and line breaks cannot hide a match. Matching is by words only, never by id: development files reuse ids
such as B04 for different questions.

The scan covers tieout/, semantics/, scripts/, finetune/ (training data), the benchmark files other than the
held-out ones, cache/llm/ (cached model calls), results/, and the development scripts and logs in ../dev_scripts.

Not scanned: version control and virtual environments, folders that are clones of other projects, the filings
themselves (data/pdf, data/words, data/sec), binary files (model weights, pickles, images, PDFs), this check's own
reports, and the held-out questions with the files derived from them. The script reads those questions itself and
prints or writes ids, file paths, counts and scores only, never question or answer text.

Calibration: run it on the 39 development questions instead. They sit in result files, cached model calls and
the learned store, so finding them there shows the detector finds a verbatim copy. The same run then tests the
two thresholds on reworded copies: each development question is reworded five mechanical ways (words dropped,
words swapped for synonyms, the end moved to the front), and the share of reworded copies flagged is measured
beside the highest score any different question reaches, which is what a false alarm would score. The report
says whether the thresholds sit between the two.

Usage (from the repo root):
    python scripts/check_leakage.py                                          # -> results/leakage_check.json
    python scripts/check_leakage.py --questions benchmark/round2/dev.jsonl   # -> results/leakage_calibration.json
Exit code 1 when anything is flagged (expected for the calibration run).
"""
from __future__ import annotations

import argparse
import codecs
import difflib
import gzip
import json
import os
import re
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import IO, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
BENCH2 = ROOT.parent / "bench2"

HELD_OUT_QUESTIONS = "benchmark/round2/questions.jsonl"
HELD_OUT = {HELD_OUT_QUESTIONS, "benchmark/round2/questions_only.jsonl", "benchmark/round2/candidates_B.jsonl",
            "benchmark/round2/candidates_C.jsonl", "benchmark/round2/adjudication.json"}
REPORT = {"held-out": "results/leakage_check.json", "calibration": "results/leakage_calibration.json"}
DEV_SCRIPTS = ROOT.parent / "dev_scripts"
OUTSIDE = ("authorA", "round2_A_questions.jsonl", "used_questions.txt")   # development material in bench2
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}
FILINGS = {"data/pdf", "data/words", "data/sec"}
LINE_CHECKED = ("finetune/", "semantics/", "benchmark/", "cache/llm/", "results/teaching_log.jsonl", "dev_scripts/",
                "bench2/round2_A_questions.jsonl", "bench2/used_questions.txt", "bench2/authorA/items")

N = 6                      # words per n-gram
CONTAINMENT_FLAG = 0.5
RATIO_FLAG = 0.85
SHORT_LINE = 400           # characters
SHARED_STEMS = 0.5         # share of a target's word stems a line needs before it is compared letter by letter
BLOCK, OVERLAP, SNIFF = 1 << 22, 1 << 16, 1 << 16   # characters per window, overlap, bytes read to spot binaries

WORD = re.compile(r"[^\W_]+")
ESCAPE = re.compile(r"\\(?:u([0-9a-fA-F]{4})|([nr])|[tbf\\\"'/])")
LINE_END = re.compile(r'[\n"]')


def unescape(text: str) -> str:
    """Undo JSON and Python string escapes, so a question stored inside a JSON string reads as plain text."""
    return ESCAPE.sub(_unescaped, text)


def _unescaped(match: re.Match) -> str:
    if match.group(1):
        return chr(int(match.group(1), 16))
    return "\n" if match.group(2) else " "


def words(text: str) -> list[str]:
    """The normalised form of a text: its lower-case runs of letters and digits."""
    return WORD.findall(text.lower())


def ngrams(tokens: list[str]) -> Iterator[tuple[str, ...]]:
    return zip(*(islice(tokens, i, None) for i in range(N)))


def stems(tokens: Iterable[str]) -> frozenset[str]:
    """The first four letters of every word of four or more: what a reworded line keeps of the original."""
    return frozenset(word[:4] for word in tokens if len(word) >= 4)


def trim(items: list, cut_start: bool, cut_end: bool) -> list:
    """Drop the first and last item of a window cut out of a longer file: they may be pieces of a word or
    a line that the neighbouring window holds whole."""
    return items[int(cut_start):len(items) - int(cut_end)]


@dataclass
class Target:
    """A question or a gold answer to look for, normalised. Its text is never printed or written."""
    id: str
    part: str
    text: str = field(repr=False)
    grams: frozenset = field(repr=False)
    stems: frozenset = field(repr=False)
    matcher: difflib.SequenceMatcher = field(repr=False)

    @classmethod
    def build(cls, id_: str, part: str, tokens: list[str]) -> Target:
        text = " ".join(tokens)
        matcher = difflib.SequenceMatcher(None, autojunk=False)
        matcher.set_seq2(text)
        return cls(id_, part, text, frozenset(ngrams(tokens)), stems(tokens), matcher)


def as_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(map(str, value))
    return "" if value is None else str(value)


def load_targets(path: Path) -> tuple[list[Target], list[str], list[dict]]:
    """The questions and gold answers of a benchmark file. A text of fewer than N words has no n-gram and is
    listed as untested instead: a bare number or name would match half the repo."""
    targets, ids, untested = [], [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            raise SystemExit(f"{path.name}: line {number} is not JSON") from None
        ids.append(record["id"])
        for part in ("question", "answer"):
            tokens = words(as_text(record.get(part)))
            if len(tokens) >= N:
                targets.append(Target.build(record["id"], part, tokens))
            else:
                untested.append({"id": record["id"], "part": part, "words": len(tokens)})
    return targets, ids, untested


@dataclass
class Source:
    """A text file to scan: where it is, its name in the report, and how to read it."""
    path: Path
    label: str
    encoding: str

    @property
    def lines_checked(self) -> bool:
        return self.label.endswith(".py") or self.label.startswith(LINE_CHECKED)


@dataclass
class Inventory:
    """What the walk met: the files to scan, and what it left out and why."""
    sources: list[Source] = field(default_factory=list)
    skipped_dirs: list[str] = field(default_factory=list)
    binary: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)


def open_file(path: Path, mode: str, encoding: str | None = None) -> IO:
    """Open a file, undoing gzip when its name ends in .gz."""
    errors = "replace" if encoding else None
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding=encoding, errors=errors)
    return open(path, mode, encoding=encoding, errors=errors)


def text_encoding(path: Path) -> str | None:
    """How to read a file as text, or None when it is binary: a NUL byte in its first 64 KB, gzip undone."""
    with open_file(path, "rb") as handle:
        head = handle.read(SNIFF)
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    return None if b"\0" in head else "utf-8-sig"


def walk(top: Path, base: Path, prefix: str, inventory: Inventory) -> Iterator[Path]:
    """The files under top, or top itself when it is a file, leaving out folders that cannot hold the
    system's own text: tooling, clones of other projects (a nested .git) and the filings."""
    if top.is_file():
        yield top
        return
    for folder, dirnames, filenames in os.walk(top):
        here = Path(folder)
        kept = []
        for name in sorted(dirnames):
            label = prefix + (here / name).relative_to(base).as_posix()
            if name in SKIP_DIRS or label in FILINGS or (here / name / ".git").exists():
                inventory.skipped_dirs.append(label)
            else:
                kept.append(name)
        dirnames[:] = kept
        yield from (here / name for name in sorted(filenames))


def collect(excluded: set[str]) -> Inventory:
    """Every text file of the repo, then the development material outside it."""
    inventory = Inventory()
    roots = ([(ROOT, ROOT, "")] + [(BENCH2 / name, BENCH2, "bench2/") for name in OUTSIDE]
             + [(DEV_SCRIPTS, DEV_SCRIPTS.parent, "")])
    for top, base, prefix in roots:
        if not top.exists():
            inventory.missing.append(prefix + top.relative_to(base).as_posix())
            continue
        for path in walk(top, base, prefix, inventory):
            label = prefix + path.relative_to(base).as_posix()
            if label in excluded:
                inventory.excluded.append(label)
                continue
            try:
                encoding = text_encoding(path)
            except (OSError, EOFError):
                inventory.unreadable.append(label)
                continue
            if encoding is None:
                inventory.binary.append(label)
            else:
                inventory.sources.append(Source(path, label, encoding))
    return inventory


def read_blocks(source: Source) -> Iterator[tuple[str, bool, bool]]:
    """A file's text in windows of BLOCK characters that overlap by OVERLAP, each with whether it was cut from
    the file at its start and at its end. Anything shorter than the overlap lies whole inside one window."""
    with open_file(source.path, "rt", source.encoding) as handle:
        block, cut_start = handle.read(BLOCK), False
        while block:
            more = handle.read(BLOCK - OVERLAP)
            yield block, cut_start, bool(more)
            block, cut_start = (block[-OVERLAP:] + more if more else ""), True


@dataclass
class FileScore:
    """How close one file comes to each target, in target order."""
    containment: list[float]
    present: list[bool]
    ratio: list[float]


class Detector:
    """Scores one file at a time against every target. Line comparisons are remembered, since the same
    training data and prompts recur in several folders."""

    def __init__(self, targets: list[Target]) -> None:
        self.targets = targets
        self.grams = frozenset().union(*(target.grams for target in targets))
        self.by_stem: dict[str, list[int]] = defaultdict(list)
        for index, target in enumerate(targets):
            for stem in target.stems:
                self.by_stem[stem].append(index)
        self.seen: dict[str, dict[int, float]] = {}

    def score(self, source: Source) -> FileScore:
        hits: set[tuple[str, ...]] = set()
        present: set[int] = set()
        pieces: set[str] = set()
        for block, cut_start, cut_end in read_blocks(source):
            text = unescape(block)
            tokens = trim(words(text), cut_start, cut_end)
            found = self.grams.intersection(ngrams(tokens))
            hits |= found
            present |= self.present_in(tokens, found, present)
            if source.lines_checked:
                pieces.update(piece for piece in trim(LINE_END.split(text), cut_start, cut_end)
                              if len(piece) < SHORT_LINE)
        return FileScore([len(target.grams & hits) / len(target.grams) for target in self.targets],
                         [index in present for index in range(len(self.targets))],
                         self.best_ratios(pieces))

    def present_in(self, tokens: list[str], found: set, known: set[int]) -> set[int]:
        """Targets whose whole text appears in this window; only those with all their n-grams found can."""
        candidates = [index for index, target in enumerate(self.targets)
                      if index not in known and target.grams <= found]
        if not candidates:
            return set()
        text = f" {' '.join(tokens)} "
        return {index for index in candidates if f" {self.targets[index].text} " in text}

    def best_ratios(self, pieces: set[str]) -> list[float]:
        best = [0.0] * len(self.targets)
        for line in {" ".join(words(piece)) for piece in pieces}:
            for index, ratio in self.compare(line).items():
                best[index] = max(best[index], ratio)
        return best

    def compare(self, line: str) -> dict[int, float]:
        """difflib ratio of a line against each target that shares at least half of its word stems. A line
        within ratio 0.85 of a text keeps far more of its words, so the filter skips no line that could flag."""
        if line in self.seen:
            return self.seen[line]
        shared = Counter(index for stem in stems(line.split()) for index in self.by_stem.get(stem, ()))
        ratios = {}
        for index, count in shared.items():
            target = self.targets[index]
            if count >= SHARED_STEMS * len(target.stems):
                target.matcher.set_seq1(line)
                ratios[index] = target.matcher.ratio()
        if ratios:
            self.seen[line] = ratios
        return ratios


class Tally:
    """Flags and per-target maxima, gathered file by file."""

    def __init__(self, targets: list[Target]) -> None:
        self.targets = targets
        self.flags: list[dict] = []
        self.best = [{"containment": 0.0, "ratio": 0.0} for _ in targets]
        self.nearest: list[dict[str, str | None]] = [{"containment": None, "ratio": None} for _ in targets]

    def add(self, label: str, score: FileScore) -> None:
        for index, target in enumerate(self.targets):
            containment, ratio = score.containment[index], score.ratio[index]
            if score.present[index]:
                self.flag(target, label, "presence", 1.0)
            if containment >= CONTAINMENT_FLAG:
                self.flag(target, label, "containment", containment)
            if ratio >= RATIO_FLAG:
                self.flag(target, label, "line", ratio)
            for measure, value in (("containment", containment), ("ratio", ratio)):
                if value > self.best[index][measure]:
                    self.best[index][measure], self.nearest[index][measure] = value, label

    def flag(self, target: Target, label: str, kind: str, score: float) -> None:
        self.flags.append({"id": target.id, "part": target.part, "file": label, "kind": kind,
                           "score": round(score, 3)})

    def per_id(self, values: list[dict]) -> dict[str, dict[str, dict]]:
        table: dict[str, dict[str, dict]] = defaultdict(dict)
        for target, value in zip(self.targets, values):
            table[target.id][target.part] = value
        return dict(table)


def area(label: str) -> str:
    """The folder a file sits in, at most two levels deep: where matches were found, for the summary."""
    return "/".join(label.split("/")[:-1][:2]) or "(repo root)"


def summarise(flags: list[dict], ids: list[str], role: str) -> tuple[dict, str]:
    """Which ids were flagged, in which folders, and the one-line verdict."""
    flagged = {part: sorted({flag["id"] for flag in flags if flag["part"] == part}) for part in ("question", "answer")}
    areas: dict[str, dict[str, set]] = defaultdict(lambda: {"question": set(), "answer": set()})
    for flag in flags:
        areas[area(flag["file"])][flag["part"]].add(flag["id"])
    summary = {"ids_flagged": flagged,
               "ids_not_flagged": sorted(set(ids) - set(flagged["question"]) - set(flagged["answer"])),
               "ids_flagged_per_area": {name: {part: len(found) for part, found in parts.items()}
                                        for name, parts in sorted(areas.items())}}
    if role == "calibration":
        verdict = (f"calibration: {len(flagged['question'])} of {len(ids)} questions and "
                   f"{len(flagged['answer'])} of {len(ids)} gold answers found in at least one file")
    elif flags:
        verdict = f"FAIL: {len(flags)} flags on {len(set(flagged['question']) | set(flagged['answer']))} held-out ids"
    else:
        verdict = "PASS: no held-out question or gold answer overlaps any scanned file"
    return summary, verdict


SYNONYMS = {"revenue": "sales", "total": "overall", "income": "earnings", "reported": "stated", "did": "does",
            "have": "hold", "what": "which", "was": "is", "year": "period", "fiscal": "financial", "company": "firm",
            "increase": "rise", "decrease": "fall", "largest": "biggest", "highest": "top", "between": "among"}
REWORDINGS = {
    "every_10th_word_dropped": lambda w: [x for i, x in enumerate(w, 1) if i % 10],
    "every_4th_word_dropped": lambda w: [x for i, x in enumerate(w, 1) if i % 4],
    "synonyms": lambda w: [SYNONYMS.get(x, x) for x in w],
    "last_4_words_moved_first": lambda w: w[-4:] + w[:-4],
    "first_3_words_replaced": lambda w: ["please", "tell", "me"] + w[3:],
}
SWEEP = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)


def flagged(containment: float, present: bool, ratio: float) -> bool:
    return present or containment >= CONTAINMENT_FLAG or ratio >= RATIO_FLAG


def reworded_matrix(targets: list[Target], kind: str) -> list[FileScore]:
    """Score every question against one file holding one reworded question, for each question in turn.
    Row j of the result: the file with question j reworded; entry j of that row is the question's own
    reworded copy, every other entry a different question."""
    detector, rows = Detector(targets), []
    with tempfile.TemporaryDirectory() as folder:
        for number, target in enumerate(targets):
            path = Path(folder) / f"{number}.txt"
            path.write_text(" ".join(REWORDINGS[kind](target.text.split())), encoding="utf-8")
            rows.append(detector.score(Source(path, "benchmark/reworded.txt", "utf-8")))
    return rows


def paraphrase_calibration(targets: list[Target]) -> dict:
    """How the two thresholds do on reworded questions, and on different questions (false alarms)."""
    questions = [target for target in targets if target.part == "question"]
    kinds: dict[str, dict] = {}
    different: list[tuple[float, bool, float]] = []
    twins: list[dict] = []
    own: list[tuple[float, bool, float]] = []
    for kind in REWORDINGS:
        diagonal = []
        for j, score in enumerate(reworded_matrix(questions, kind)):
            for i in range(len(questions)):
                entry = (score.containment[i], score.present[i], score.ratio[i])
                (diagonal if i == j else different).append(entry)
                if i != j and flagged(*entry):
                    twins.append({"reworded": questions[j].id, "matches": questions[i].id, "how": kind,
                                  "containment": round(entry[0], 3), "ratio": round(entry[2], 3)})
        own += diagonal
        kinds[kind] = {"n": len(diagonal), "flagged": sum(flagged(*entry) for entry in diagonal),
                       "median_containment": round(statistics.median(e[0] for e in diagonal), 3),
                       "median_ratio": round(statistics.median(e[2] for e in diagonal), 3),
                       "lowest_ratio": round(min(e[2] for e in diagonal), 3)}
    false_alarms = sum(flagged(*entry) for entry in different)
    return {
        "reworded_copies_flagged": kinds,
        "different_questions": {"pairs": len(different), "flagged": false_alarms,
                                "highest_containment": round(max(e[0] for e in different), 3),
                                "highest_ratio": round(max(e[2] for e in different), 3), "flagged_pairs": twins},
        "ratio_threshold_sweep": {str(t): {"reworded_copies_caught": sum(e[2] >= t for e in own),
                                           "of": len(own), "false_alarms": sum(e[2] >= t for e in different)}
                                  for t in SWEEP},
        "verdict": ("thresholds separate reworded copies from different questions: no false alarm" if not false_alarms
                    else f"{false_alarms} false alarms: a threshold is too low"),
    }


def resolve(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def label_of(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Look for the held-out questions in everything that shaped the system.")
    parser.add_argument("--questions", default=HELD_OUT_QUESTIONS, help="benchmark file whose questions to look for")
    parser.add_argument("--out", help="report path (default: by role, see REPORT)")
    return parser.parse_args(argv)


def report(args: argparse.Namespace, started: float) -> tuple[dict, Path]:
    questions = resolve(args.questions)
    role = "held-out" if questions.resolve() == (ROOT / HELD_OUT_QUESTIONS).resolve() else "calibration"
    out = resolve(args.out or REPORT[role])
    targets, ids, untested = load_targets(questions)
    inventory = collect(HELD_OUT | set(REPORT.values()) | {label_of(questions), label_of(out)})
    detector, tally = Detector(targets), Tally(targets)
    for source in inventory.sources:
        tally.add(source.label, detector.score(source))
    summary, verdict = summarise(tally.flags, ids, role)
    paraphrases = {"paraphrase_calibration": paraphrase_calibration(targets)} if role == "calibration" else {}
    rounded = [{measure: round(value, 3) for measure, value in best.items()} for best in tally.best]
    return {
        "verdict": verdict,
        "role": role,
        "questions_file": label_of(questions),
        "method": {"ngram_words": N, "containment_flag": CONTAINMENT_FLAG, "ratio_flag": RATIO_FLAG,
                   "short_line_chars": SHORT_LINE, "line_needs_shared_stems": SHARED_STEMS,
                   "line_checked_prefixes": list(LINE_CHECKED) + ["*.py"]},
        "seconds": round(time.perf_counter() - started, 1),
        "held_out_ids" if role == "held-out" else "calibration_ids": ids,
        "untested": untested,
        **paraphrases,
        "summary": summary,
        "flags": tally.flags,
        "max_score_per_id": tally.per_id(rounded),
        "nearest_file_per_id": tally.per_id(tally.nearest),
        "checked_files": [source.label for source in inventory.sources],
        "skipped": {"dirs": inventory.skipped_dirs, "binary_files": inventory.binary,
                    "held_out_and_own_reports": inventory.excluded, "missing": inventory.missing,
                    "unreadable": inventory.unreadable},
    }, out


def show(result: dict, out: Path) -> None:
    """Print the verdict and counts: ids, paths and scores only."""
    print(result["verdict"])
    print(f"files checked {len(result['checked_files'])}, binary skipped {len(result['skipped']['binary_files'])}, "
          f"folders skipped {len(result['skipped']['dirs'])}, seconds {result['seconds']}")
    for name in ("missing", "unreadable"):
        if result["skipped"][name]:
            print(f"{name}: {', '.join(result['skipped'][name])}")
    print("untested (fewer than 6 words):", " ".join(f"{u['id']}/{u['part']}" for u in result["untested"]) or "none")
    if "paraphrase_calibration" in result:
        calibration = result["paraphrase_calibration"]
        print("reworded copies flagged:", {k: f"{v['flagged']}/{v['n']}" for k, v in calibration["reworded_copies_flagged"].items()})
        print("different questions:", {k: v for k, v in calibration["different_questions"].items() if k != "flagged_pairs"},
              "|", calibration["verdict"])
        print("  flagged pairs:", " ".join(f"{t['reworded']}~{t['matches']}({t['how'][:6]})" for t in calibration["different_questions"]["flagged_pairs"]))
    for name, parts in result["summary"]["ids_flagged_per_area"].items():
        print(f"  {name:32} ids flagged: question {parts['question']:3}  answer {parts['answer']:3}")
    print("ids not flagged:", " ".join(result["summary"]["ids_not_flagged"]) or "none")
    if result["role"] == "held-out":
        for flag in result["flags"][:100]:
            print(f"  FLAG {flag['id']} {flag['part']} {flag['kind']} {flag['score']} {flag['file']}")
    print("report:", label_of(out))


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    result, out = report(parse_args(argv), started)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    show(result, out)
    return 1 if result["flags"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
