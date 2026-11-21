"""Read text whose font has lost its character map.

Some PDFs embed subset fonts without a ToUnicode table. pdfplumber then
reports each glyph as "(cid:N)", where N is only the glyph's number inside
the embedded font, so the text layer is a substitution cipher. Two facts break
the cipher without OCR:

1. Order. Font subsetters number the glyphs they keep in increasing order of
   the characters they stand for, so the map from glyph codes to characters
   is strictly increasing (with gaps for characters the font never needed).
2. Width. Every glyph's advance width is observable, and the common office
   fonts are metric twins of the standard PDF fonts whose widths pdfminer
   ships (Arial = Helvetica, Times New Roman = Times, Courier New = Courier).

So decoding is a sequence alignment: map the sorted codes onto the sorted
reference alphabet, strictly increasing, so that widths agree. Dynamic
programming finds every alignment that matches the most glyph occurrences.
A code with one possible character is settled by width and order alone.
Where several characters remain possible (runs of equal-width characters
some of which the font does not use), words from other documents act as a
crib: the alignment that turns the most cipher words into known words wins,
and only by a clear margin. A caller may add a further judge (for digits,
statement arithmetic). Whatever is still open stays undecoded, is shown as
U+FFFD, and is counted.

The unit of decoding is one embedded font program. Several pages often share
one program behind per-page wrappers, which pools the evidence.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from pdfminer.fontmetrics import FONT_METRICS
from pdfminer.pdftypes import resolve1
from pdfminer.psparser import PSLiteral

CID = re.compile(r"^\(cid:(\d+)\)$")
UNKNOWN = "�"
TOLERANCE = 2            # width agreement, in 1/1000 em
BIG = 10 ** 6            # width evidence always outranks vocabulary evidence
MARGIN = 3.0             # the crib must win by this much (log2 word-frequency units)
MAX_COMBOS = 64          # wildcard expansions tried per cipher word
FAMILIES = {"Helvetica": ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"),
            "Times": ("Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"),
            "Courier": ("Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique")}


@dataclass
class FontUnit:
    """One embedded font program: its codes, their widths and how often each occurs."""
    key: tuple
    name: str
    widths: dict = field(default_factory=dict)       # code -> Counter of observed widths
    counts: Counter = field(default_factory=Counter)  # code -> occurrences
    runs: Counter = field(default_factory=Counter)   # tuple of touching codes -> occurrences
    reference: str = ""
    candidates: dict = field(default_factory=dict)   # code -> set of possible characters
    decoded: dict = field(default_factory=dict)      # code -> (character, evidence)


def cid_of(text: str) -> Optional[int]:
    m = CID.match(text)
    return int(m.group(1)) if m else None


def font_keys(page) -> dict:
    """fontname -> identity of the embedded font program behind it on this page.

    A composite font's identity is its descendant font object, which pages
    share even when each page has its own wrapper object.
    """
    out: dict = {}
    try:
        fonts = resolve1((page.page_obj.resources or {}).get("Font")) or {}
    except Exception:
        return out
    for ref in fonts.values():
        spec = resolve1(ref) or {}
        base = spec.get("BaseFont")
        name = base.name if isinstance(base, PSLiteral) else str(base)
        target = resolve1(spec.get("DescendantFonts")) or [ref]
        objid = getattr(target[0], "objid", None)
        if objid is None or out.get(name, objid) != objid:
            out[name] = None                      # unknown or clashing: stay page-local
        else:
            out[name] = ("obj", objid)
    return out


def unit_key(keys: dict, page: int, fontname: str) -> tuple:
    return keys.get(fontname) or ("page", page, fontname)


def collect(pages: dict, keys: dict) -> dict:
    """Group cipher glyphs into font units. pages maps a page index to its
    pdfplumber chars, keys maps it to that page's font_keys.
    Returns unit key -> FontUnit."""
    units: dict = {}
    for i, chars in pages.items():
        run: list = []
        prev = None
        for ch in chars + [None]:
            code = cid_of(ch["text"]) if ch else None
            key = unit_key(keys[i], i, ch["fontname"]) if code is not None else None
            if run and (key != run_key or not _adjacent(prev, ch)):
                units[run_key].runs[tuple(run)] += 1
                run = []
            if key is None:
                continue
            unit = units.setdefault(key, FontUnit(key, ch["fontname"]))
            unit.widths.setdefault(code, Counter())[round(ch["adv"] * 1000)] += 1
            unit.counts[code] += 1
            run.append(code)
            run_key, prev = key, ch
    return units


def _adjacent(a: Optional[dict], b: Optional[dict]) -> bool:
    """True when glyph b continues glyph a on the same line with no visible gap."""
    if a is None or b is None:
        return False
    slack = 0.15 * a["size"]
    return abs(a["top"] - b["top"]) < slack and -slack < b["x0"] - a["x1"] < slack


def references(fontname: str) -> list:
    """Standard metric sets to try, the one the font name suggests first."""
    low = fontname.lower()
    family = "Times" if "times" in low or "roman" in low else "Courier" if "courier" in low else "Helvetica"
    style = (2 if "italic" in low or "oblique" in low else 0) + (1 if "bold" in low else 0)
    hint = FAMILIES[family][style]
    return [hint] + [f for fam in FAMILIES.values() for f in fam if f != hint]


def alphabet(reference: str) -> list:
    """(character, width) of a standard font, in increasing character order."""
    widths = FONT_METRICS[reference][1]
    return sorted(((c, w) for c, w in widths.items() if len(c) == 1 and not 0xE000 <= ord(c) <= 0xF8FF),
                  key=lambda cw: ord(cw[0]))


def align(codes: list, weights: list, alpha: list, score: Callable[[int, int], float],
          allowed: Optional[list] = None) -> tuple:
    """Monotone alignment of codes onto the alphabet.

    codes[i] is the observed width of the i-th code (codes sorted), weights[i]
    its occurrences; score(i, j) adds evidence for code i = alpha[j]; allowed[i],
    if given, limits code i to those characters. A code may stay unmatched.
    Returns (best total, totals) where totals[i] maps each alphabet index j to
    the best total of any alignment with code i -> alpha[j].
    """
    n, m = len(codes), len(alpha)
    ok = [[abs(codes[i] - alpha[j][1]) <= TOLERANCE and (allowed is None or alpha[j][0] in allowed[i])
           for j in range(m)] for i in range(n)]
    NEG = float("-inf")
    fwd = [[0.0] * (m + 1) for _ in range(n + 1)]        # codes[:i] within alpha[:j]
    for i in range(1, n + 1):
        fwd[i][0] = fwd[i - 1][0]
        for j in range(1, m + 1):
            v = max(fwd[i][j - 1], fwd[i - 1][j])
            if ok[i - 1][j - 1]:
                v = max(v, fwd[i - 1][j - 1] + weights[i - 1] * BIG + score(i - 1, j - 1))
            fwd[i][j] = v
    bwd = [[0.0] * (m + 1) for _ in range(n + 1)]        # codes[i:] within alpha[j:]
    for i in range(n - 1, -1, -1):
        bwd[i][m] = bwd[i + 1][m]
        for j in range(m - 1, -1, -1):
            v = max(bwd[i][j + 1], bwd[i + 1][j])
            if ok[i][j]:
                v = max(v, bwd[i + 1][j + 1] + weights[i] * BIG + score(i, j))
            bwd[i][j] = v
    totals = [{j: fwd[i][j] + weights[i] * BIG + score(i, j) + bwd[i + 1][j + 1]
               for j in range(m) if ok[i][j]} for i in range(n)]
    return fwd[n][m], [{j: t for j, t in row.items() if t > NEG} for row in totals]


def _width(unit: FontUnit, code: int) -> int:
    return unit.widths[code].most_common(1)[0][0]


def spans(run: tuple, blank: set) -> list:
    """(start, end) of the words in a run of codes, cut at blank codes."""
    out, start = [], 0
    for k, c in enumerate(run + (None,)):
        if c is None or c in blank:
            if k > start:
                out.append((start, k))
            start = k + 1
    return out


def word_hits(unit: FontUnit, vocab: Counter) -> Callable[[int, str], float]:
    """Evidence for code = char from the known words it produces.

    A letter scores the words that contain it; a white-space character scores
    the words it separates. Other open codes range over their candidates and
    the most common known reading counts. Each cipher word occurrence adds
    log2(1 + how often the crib prints that word), so "2025" outweighs "1012".
    """
    blank = {c for c, (ch, _) in unit.decoded.items() if ch.isspace()}
    by_code = defaultdict(list)
    for run, n in unit.runs.items():
        for c in set(run):
            by_code[c].append((run, n))

    def known(word: tuple, code: int, char: str) -> int:
        options = [[char] if c == code else sorted(unit.candidates.get(c) or {UNKNOWN}) for c in word]
        combos = 1
        for o in options:
            combos *= len(o)
        if combos > MAX_COMBOS:
            return 0
        return max(vocab.get("".join(p), 0) for p in _product(options))

    def hits(code: int, char: str) -> float:
        space = char.isspace()
        total = 0.0
        for run, n in by_code[code]:
            for s, e in spans(run, blank | {code} if space else blank):
                touches = (e < len(run) and run[e] == code) or (s > 0 and run[s - 1] == code)
                if touches if space else code in run[s:e]:
                    total += n * math.log2(1 + known(run[s:e], code, char))
        return total
    return hits


def _product(options: list) -> Iterable:
    if not options:
        yield ()
        return
    for head in options[0]:
        for tail in _product(options[1:]):
            yield (head,) + tail


def _table(unit: FontUnit) -> tuple:
    codes = sorted(unit.counts)
    return codes, [_width(unit, c) for c in codes], [unit.counts[c] for c in codes]


def narrow(unit: FontUnit, evidence: str) -> None:
    """Candidates = characters some best alignment gives, within the current
    candidates; a code left with one candidate is settled by `evidence`."""
    codes, widths, weights = _table(unit)
    alpha = alphabet(unit.reference)
    allowed = [unit.candidates.get(c) for c in codes] if unit.candidates else None
    total, totals = align(widths, weights, alpha, lambda i, j: 0, allowed)
    for i, c in enumerate(codes):
        unit.candidates[c] = {alpha[j][0] for j, t in totals[i].items() if t == total}
        if len(unit.candidates[c]) == 1 and c not in unit.decoded:
            unit.decoded[c] = (next(iter(unit.candidates[c])), evidence)


def open_codes(unit: FontUnit) -> list:
    return [c for c in sorted(unit.counts) if len(unit.candidates.get(c, ())) > 1]


def solve(unit: FontUnit, vocab: Counter) -> None:
    """Choose the reference metrics, then settle codes by width and order, then by the crib."""
    codes, widths, weights = _table(unit)
    best = None
    for ref in references(unit.name):
        total = align(widths, weights, alphabet(ref), lambda i, j: 0)[0]
        if best is None or total > best[0]:
            best = (total, ref)
    unit.reference = best[1]
    alpha = alphabet(unit.reference)
    narrow(unit, "width")
    spaces_first = True                           # words exist only once the spaces are known
    while vocab:                                  # crib rounds until nothing changes
        open_ = [i for i, c in enumerate(codes) if len(unit.candidates[c]) > 1]
        if not open_:
            break
        spaces = [i for i in open_ if any(ch.isspace() for ch in unit.candidates[codes[i]])]
        decide = spaces if spaces and spaces_first else open_
        hits = word_hits(unit, vocab)
        index = {ch: j for j, (ch, _) in enumerate(alpha)}
        cache = {(i, index[ch]): hits(codes[i], ch) for i in open_ for ch in unit.candidates[codes[i]]}
        allowed = [unit.candidates[c] for c in codes]
        _, scored = align(widths, weights, alpha, lambda i, j: cache.get((i, j), 0), allowed)
        changed = False
        for i in decide:
            ranked = sorted(scored[i].values(), reverse=True)
            if len(ranked) > 1 and ranked[0] - ranked[1] >= MARGIN:
                j = max(scored[i], key=scored[i].get)
                unit.candidates[codes[i]] = {alpha[j][0]}
                changed = True
        if not changed and decide is spaces and spaces_first:
            spaces_first = False
            continue
        if not changed:
            break
        narrow(unit, "vocabulary")


def monotone_choices(options: list, limit: int = 5040) -> list:
    """Strictly increasing picks, one character from each option set, at most limit of them."""
    out: list = []

    def walk(k: int, prefix: tuple) -> None:
        if len(out) >= limit:
            return
        if k == len(options):
            out.append(prefix)
            return
        for ch in sorted(options[k]):
            if not prefix or ord(ch) > ord(prefix[-1]):
                walk(k + 1, prefix + (ch,))
    walk(0, ())
    return out


def settle_by(unit: FontUnit, judge: Callable[[dict], float], codes: list) -> None:
    """Let an outside judge score every consistent reading of `codes`. A code is
    settled when all top-scoring readings agree on it and the judge told at
    least two readings apart; otherwise it stays open."""
    scored = [(judge(dict(zip(codes, pick))), pick)
              for pick in monotone_choices([unit.candidates[c] for c in codes])]
    if len(scored) < 2 or max(s for s, _ in scored) == min(s for s, _ in scored):
        return
    top = max(s for s, _ in scored)
    picks = [p for s, p in scored if s == top]
    for k, c in enumerate(codes):
        if len({p[k] for p in picks}) == 1:
            unit.candidates[c] = {picks[0][k]}
    narrow(unit, "arithmetic")


def translate(text: str, unit: Optional[FontUnit], trial: Optional[dict] = None) -> str:
    """Decoded text of one pdfplumber char; trial overrides some codes of `unit`."""
    code = cid_of(text)
    if code is None or unit is None:
        return text
    if trial and code in trial:
        return trial[code]
    return unit.decoded.get(code, (UNKNOWN, ""))[0]


def decode_chars(chars: list, keys: dict, page: int, units: dict,
                 unit: Optional[FontUnit] = None, trial: Optional[dict] = None) -> list:
    """Copies of a page's chars with decoded text."""
    out = []
    for ch in chars:
        u = units.get(unit_key(keys, page, ch["fontname"]))
        out.append(dict(ch, text=translate(ch["text"], u, trial if u is unit else None)))
    return out


def decode_pages(pages: dict, keys: dict, vocab: Counter, judge: Optional[Callable] = None) -> tuple:
    """Decode every cipher glyph of a document.

    pages: page index -> pdfplumber chars; keys: page index -> font_keys(page).
    judge(units, unit, open codes) may return (codes, scorer) for settle_by:
    the caller's own evidence (statement arithmetic, for digits).
    Returns (page index -> chars with decoded text, unit key -> FontUnit).
    """
    units = collect(pages, keys)
    for unit in sorted(units.values(), key=lambda u: -sum(u.counts.values())):
        solve(unit, vocab)
    if judge is not None:
        for unit in units.values():
            codes = open_codes(unit)
            verdict = judge(units, unit, codes) if codes else None
            if verdict:
                settle_by(unit, verdict[1], verdict[0])
    out = {i: decode_chars(chars, keys[i], i, units) for i, chars in pages.items()}
    return out, units


def summary(units: dict) -> dict:
    """Glyph occurrences by the evidence that settled them, overall and per font."""
    kinds = ("width", "vocabulary", "arithmetic", "undecoded")
    out = {"font_units": len(units), "glyph_codes": 0, "occurrences": 0, **{k: 0 for k in kinds}, "fonts": []}
    for u in sorted(units.values(), key=lambda u: -sum(u.counts.values())):
        row = {"font": u.name, "metrics": u.reference, "codes": len(u.counts),
               "occurrences": sum(u.counts.values()), **{k: 0 for k in kinds}}
        for c, n in u.counts.items():
            row[u.decoded[c][1] if c in u.decoded else "undecoded"] += n
        row["undecoded_codes"] = sum(1 for c in u.counts if c not in u.decoded)
        out["fonts"].append(row)
        out["glyph_codes"] += row["codes"]
        out["occurrences"] += row["occurrences"]
        for k in kinds:
            out[k] += row[k]
    out["share"] = {k: round(out[k] / out["occurrences"], 5) for k in kinds} if out["occurrences"] else {}
    return out
