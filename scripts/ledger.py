"""The ledger: every number a page prints is either a node of the graph or a listed gap.

The page text (`bench2/pdf_text/<DOC>/pNNN.txt`, the text the blind question authors read) is split
into number tokens. Each token is one of

  excluded    not a quantity: a year, the day of a date, a footnote marker, a page number, a
              reference ("Note 4", "Item 7", "Level 3"), an identifier (a zip code), an enumerator;
  cell        a value cell of a raw row on the same page (the three statements are raw rows too),
  sentence    a number of a numeric sentence (graph/mentions.json.gz) on the same page,
  fact        a cover-page fact (data/textfacts.json) on the same page,
  label       a number printed inside a row's label, section or title ("Notes due 2031", "4.25%"),
  residue     none of these: a printed number the graph does not hold.

Tokens are matched by value as printed (sign, "$", "%" and commas aside), line by line: a table line
is first matched against the one row that holds most of its figures, a prose line against the
sentences it is part of, and only then against everything else on the page, so that a figure printed
twice is charged to the right line. Residue lines go to graph/residue.json.gz with the lines above and
below and a class:

  split_row     the figures stand on a line of their own under a label line that has none
                (Home Depot FY2025 p37: "Average ticket (2) (3)" / "$90.56 $89.31 $90.07 1.4 % (0.8) %"),
  partial_row   a row holds some of the line's figures but lost the others (a shifted or lost cell),
  one_line_row  a label followed by figures that no row holds,
  prose         numbers inside running text that no sentence holds,
  orphan        anything else (a figure line with no label above it).

Limits, stated: the ledger reads the same text layer as the build, so a line the text layer lacks is
invisible to both; a match by value can be a coincidence (`--control` measures how often: each page is
matched against another page's nodes); a dash (a printed nil) is not counted as a number.

    python scripts/ledger.py                         # all filings: graph/residue.json.gz, results/ledger.csv
    python scripts/ledger.py --ticker HD --page 37   # the certificate of one page (any filing of HD)
    python scripts/ledger.py --doc HD_FY2025_10K --page 37
    python scripts/ledger.py --control               # coincidence rate: pages matched against page + 7
"""
import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raw_coverage as rc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NUM_RE = re.compile(r"^(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
FIND_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
MONTH_RE = re.compile(r"^(?:january|february|march|april|may|june|july|august|september|october|november|december|"
                      r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?,?$", re.I)
REF_WORDS = {"item", "items", "note", "notes", "part", "parts", "section", "sections", "rule", "rules", "form",
             "exhibit", "exhibits", "schedule", "schedules", "topic", "asc", "asu", "regulation", "article",
             "articles", "chapter", "page", "pages", "level", "levels", "tier", "class", "series", "phase",
             "no", "nos", "no.", "stage", "stages", "basel", "pillar", "appendix", "table", "figure",
             "footnote", "footnotes", "step", "tranche", "act", "code", "ifrs", "sfas", "fas", "box", "line",
             "lines", "annex", "volume", "issue", "subpart", "paragraph", "clause", "release", "proposal",
             "amendment", "chapters", "ias", "gaap", "sab", "eitf", "fin", "pcaob", "as", "w-", "rev."}
CHAIN = {"and", "or", "to", "through", "thru", ",", "&", "-", "–"}
FORM_RE = re.compile(r"\b(?:10-K|10-Q|8-K|S-8|S-3|S-4|11-K|20-F|DEF\s+14A|10-K405)\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
WORD_RE = re.compile(r"[A-Za-z]{2,}")
DASH_TOKENS = {"—", "–", "-", "$—", "$-", "$–", "—%", "-%", "(—)", "—)"}


def num_norm(text: str) -> str:
    """The digits of the first number in a text, commas removed: '$(1,234.5)' -> '1234.5'."""
    m = FIND_NUM.search(text or "")
    if not m:
        return ""
    v = m.group(0).rstrip(",").replace(",", "")
    if "." in v:
        head, _, frac = v.partition(".")
        v = (head.lstrip("0") or "0") + "." + frac
    else:
        v = v.lstrip("0") or "0"
    return v


def split_token(tok: str):
    """(number text, dollar, negative, percent, parenthesised) when tok is a printed number, else None."""
    t = tok.strip("\"“”'‘’[]{}*†‡")
    t = t.rstrip(",;:")
    if t.endswith(".") and not t.endswith(".."):
        t = t[:-1]
    t = t.rstrip(",;:")
    neg = False
    paren = False
    dollar = False
    if t.startswith("$("):
        t, dollar, neg, paren = t[2:], True, True, True
    if t.startswith("("):
        t, paren = t[1:], True
    if t.startswith("$"):
        t, dollar = t[1:], True
    if t.startswith(("-", "−", "+")):
        neg = t[0] != "+"
        t = t[1:]
    if t.startswith("$"):
        t, dollar = t[1:], True
    pct = False
    for _ in range(3):
        if t.endswith("%"):
            t, pct = t[:-1], True
        elif t.endswith(")"):
            t = t[:-1]
            if paren:
                neg = True
        elif t.endswith(("x", "X")) and NUM_RE.match(t[:-1] or "-"):
            t = t[:-1]                                  # "1.2x" coverage ratios
    if paren and not tok.rstrip(",;:.%").endswith(")"):
        return None                                     # "(Note" or "(in" opened a parenthetical
    if not NUM_RE.match(t):
        return None
    return t, dollar, neg, pct, paren


def tokens_of(line: str, page_markers: set, first_or_last: bool) -> list:
    """Number tokens of one line, each with its kind of exclusion ('' when it is a quantity)."""
    toks = line.split()
    out = []
    prev_word = ""
    prev_kind = ""
    exhibit_line = bool(FORM_RE.search(line))
    for i, tok in enumerate(toks):
        sp = split_token(tok)
        if sp is None:
            if tok in DASH_TOKENS or tok.strip("$%()") in ("—", "–"):
                prev_kind = "dash"
            elif tok not in ("$", "%", "*", "†"):
                low = tok.lower().strip("(),.;:")
                if low in CHAIN and prev_kind == "ref":
                    continue                            # "Notes 3 and 4", "Levels 1, 2 and 3"
                prev_kind = "word" if WORD_RE.search(tok) else prev_kind
                prev_word = low
            continue
        text, dollar, neg, pct, paren = sp
        if i + 1 < len(toks) and toks[i + 1].startswith("%"):
            pct = True
        bare = not (dollar or pct or "," in text or "." in text)
        val = float(text.replace(",", ""))
        excl = ""
        if bare and paren and val <= 40 and val == int(val) and \
                (i == 0 or prev_kind in ("word", "marker", "ref") and (int(val) in page_markers or
                                                                     i + 1 < len(toks) or prev_kind == "marker")):
            excl = "marker"                             # "(2)" after a label, "(1) Does not include ..."
        elif bare and i == 0 and len(toks) > 1 and re.match(r"^\d{1,2}[.)]$", tok):
            excl = "enumerator"                         # "1." opening a list item
        elif bare and 1800 <= val <= 2099 and len(text) == 4:
            excl = "year"                               # "fiscal 2025", "the Exchange Act of 1934"
        elif bare and 1 <= val <= 31 and i > 0 and MONTH_RE.match(toks[i - 1]):
            excl = "date"                               # "June 18, 2024"
        elif not (dollar or pct) and (prev_word in REF_WORDS or prev_kind == "ref" and bare):
            excl = "reference"                          # "Note 4", "Item 7", "Section 9.14", "Notes 3 and 4"
        elif exhibit_line and not (dollar or pct) and re.fullmatch(r"\d{1,3}(?:\.\d{1,3})?", text):
            excl = "reference"                          # an exhibit index line: "10.07* ... 10-Q 001-35992 10.09"
        elif bare and first_or_last and val <= 1500 and (i == len(toks) - 1 or i == 0):
            excl = "page"                               # "Fiscal 2025 Form 10-K 29", "33 Bank of America"
        elif not (dollar or pct) and i == 0 and len(toks) > 1 and re.fullmatch(r"\d{1,2}\.\d{1,2}", text) and \
                toks[1][:1].isupper():
            excl = "enumerator"                         # a section number: "9.12 Governing Law.", "4.33 Form of Note"
        elif bare and len(text) >= 5:
            excl = "identifier"                         # zip code, file number, CIK
        out.append({"i": i, "text": tok, "norm": num_norm(text), "value": -val if neg else val,
                    "pct": pct, "dollar": dollar, "excl": excl})
        prev_kind = "ref" if excl == "reference" else ("marker" if excl == "marker" else "num")
    return out


def page_markers(lines: list) -> set:
    """Footnote numbers the page defines: a line that opens with "(1)" or "1 " followed by words."""
    found = set()
    for ln in lines:
        m = re.match(r"^\s*\((\d{1,2})\)\s+[A-Za-z]", ln) or re.match(r"^\s*(\d{1,2})\s+[A-Z][a-z]", ln)
        if m:
            found.add(int(m.group(1)))
    return found


def _collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


SIGNATURES_RE = re.compile(r"^\s*SIGNATURES?\s*$", re.I)
EXHIBIT_LINE_RE = re.compile(r"incorporated\s+(?:herein\s+)?by\s+reference|^\s*\*?\d{1,3}\.\d{1,3}[*†#]*\s+[A-Z]", re.I)


def page_zone(lines: list) -> str:
    """'toc' for a table of contents or an index (many lines that end in a page number, or dot leaders),
    'exhibit' for an exhibit index (lines that name forms, filings and exhibit numbers), else 'body'."""
    ends = sum(1 for ln in lines if re.search(r"[A-Za-z].*\s\d{1,3}\s*$", ln) and len(FIND_NUM.findall(ln)) == 1)
    leaders = sum(1 for ln in lines if "....." in ln)
    if ends >= 6 and ends >= 0.3 * max(1, sum(1 for ln in lines if ln.strip())) or leaders >= 3:
        return "toc"
    exhibit = sum(1 for ln in lines if FORM_RE.search(ln) or EXHIBIT_LINE_RE.search(ln))
    return "exhibit" if exhibit >= 4 else "body"


def zones_of(files: list) -> dict:
    """page -> zone for one filing; every page after the signatures page is an exhibit (plan documents,
    indentures, certifications filed with the report)."""
    texts = {int(f.stem[1:]): f.read_text(encoding="utf-8", errors="replace").splitlines() for f in files}
    pages = sorted(texts)
    sig = [p for p in pages if any(SIGNATURES_RE.match(ln) for ln in texts[p][:12])]
    sig = [p for p in sig if pages and p >= pages[int(0.6 * (len(pages) - 1))]]
    last_sig = sig[-1] if sig else None
    return {p: ("exhibit" if last_sig is not None and p > last_sig else page_zone(texts[p])) for p in pages}


NOT_FIGURES = {"N/A", "n/a", "NA", "NM", "nm", "N/M", "n/m", "*", "**", "NMF", "nmf"}


def tail(line: str):
    """raw_coverage.tail, with "N/A" and "NM" cells at the end of the line passed over."""
    toks = line.split()
    while toks and toks[-1] in NOT_FIGURES:
        toks.pop()
    return rc.tail(" ".join(toks))


class PageNodes:
    """What the graph holds for one page, as multisets of printed numbers that can be consumed."""

    def __init__(self, rows: list, mentions: list, facts: list):
        self.rows = []
        for r in rows:
            have = Counter(num_norm(c.get("text") or "") for c in r["cells"])
            have.pop("", None)
            self.rows.append([r, have])
        self.mentions = []
        for m in mentions:
            have = Counter(num_norm(n.get("text") or "") for n in m["numbers"])
            have.pop("", None)
            self.mentions.append([m, have, _collapse(m.get("sentence", ""))])
        self.facts = Counter()
        for f in facts:
            for piece in (str(f.get("value", "")), f.get("quote", "")):
                for x in FIND_NUM.findall(piece or ""):
                    self.facts[num_norm(x)] += 1
        self.labels = Counter()
        for r in rows:
            for piece in (r.get("label", ""), r.get("section", ""), r.get("title", "")):
                for x in FIND_NUM.findall(piece or ""):
                    self.labels[num_norm(x)] += 1

    def best_row(self, want: Counter):
        best, best_n = None, 0
        for entry in self.rows:
            have = entry[1]
            n = sum(min(k, have[f]) for f, k in want.items() if have.get(f))
            if n > best_n:
                best, best_n = entry, n
        return best, best_n

    def take_from_pool(self, norm: str) -> str:
        for entry in self.rows:
            if entry[1].get(norm):
                entry[1][norm] -= 1
                return "cell"
        for entry in self.mentions:
            if entry[1].get(norm):
                entry[1][norm] -= 1
                return "sentence"
        if self.facts.get(norm):
            self.facts[norm] -= 1
            return "fact"
        if self.labels.get(norm):
            self.labels[norm] -= 1
            return "label"
        return ""


def _half_unit(text: str) -> float:
    """Half a unit of the last printed digit: "18.6" -> 0.05, "33 %" -> 0.5."""
    t = num_norm(text)
    return 0.5 * 10 ** -(len(t.split(".")[1]) if "." in t else 0)


def mark_derivable(per_line: list, nodes) -> None:
    """A residue percentage that one division of two figures the graph holds reproduces at its printed precision:
    a share of a figure on the page ("Gross profit 54,865 33.3 %": 54,865 / net sales 164,683) or the change
    between two figures of its own line ("1,601.5 1,637.2 ... (2.2) %"). The build drops such columns on purpose;
    they stay in the residue, marked, so the certificate can say how many gaps are one division away. `nodes`
    gives the page's figures (another page's in the control)."""
    page_vals = sorted({float(num_norm(c.get("text") or "")) for entry in nodes.rows for c in entry[0]["cells"]
                        if num_norm(c.get("text") or "") and float(num_norm(c.get("text") or "")) >= 10
                        and not c.get("unit") and entry[0].get("unit") in ("USD", "count", "shares")})
    for toks in per_line:
        held = [t["value"] for t in toks if t["acct"] in ("cell", "sentence") and not t["pct"] and t["value"]]
        for t in toks:
            if t["acct"] != "residue" or not (t["pct"] or "." in num_norm(t["text"])):
                continue
            v, half = t["value"], _half_unit(t["text"]) + 1e-9
            if not 0.05 <= abs(v) <= 1000:
                continue
            kind = ""
            for k, a in enumerate(held):                 # the change between two figures of the line
                for j, b in enumerate(held):
                    if k != j and b and abs(100 * (a - b) / abs(b) - v) <= half:
                        kind = "change"
                        break
                if kind:
                    break
            if not kind:
                for a in held:                           # a share of a figure the page prints (printed values)
                    a_abs = abs(a)
                    if any(b != a_abs and abs(100 * a_abs / b - abs(v)) <= half for b in page_vals if b >= a_abs * 0.5):
                        kind = "share"
                        break
            if kind:
                t["derived"] = kind


def account_page(lines: list, nodes: PageNodes, prev_nodes=None, derive_from=None) -> tuple:
    """Every number token of the page with its account; the residue lines with their class."""
    markers = page_markers(lines)
    nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
    edge = set(nonempty[:1] + nonempty[-1:])
    per_line = []
    for i, ln in enumerate(lines):
        toks = tokens_of(ln, markers, i in edge)
        for t in toks:
            t["acct"] = t["excl"] and "excluded"
        per_line.append(toks)
    row_of_line = {}
    # pass 1: a table line against the one row that holds most of its trailing figures
    for i, ln in enumerate(lines):
        qty = [t for t in per_line[i] if not t["acct"]]
        if not qty:
            continue
        figs, _ = tail(ln)
        tail_norms = Counter(num_norm(f) for f, _ in figs if num_norm(f))
        want = Counter()
        for t in qty:
            if tail_norms.get(t["norm"], 0) > want[t["norm"]]:
                want[t["norm"]] += 1
        if not want:
            continue
        entry, hit = nodes.best_row(want)
        if entry is None or hit * 2 < sum(want.values()) and hit < 2:
            continue
        row_of_line[i] = (entry[0]["id"], hit, sum(want.values()))
        for t in qty:
            if want.get(t["norm"]) and entry[1].get(t["norm"]):
                entry[1][t["norm"]] -= 1
                want[t["norm"]] -= 1
                t["acct"] = "cell"
    # pass 2: a prose line against the sentences it is part of
    for i, ln in enumerate(lines):
        qty = [t for t in per_line[i] if not t["acct"]]
        if not qty:
            continue
        key = _collapse(ln)
        if len(key) < 12:
            continue
        heads = (key, key[:28], key[-28:])
        for entry in nodes.mentions:
            sent = entry[2]
            if not any(h and h in sent for h in heads):
                continue
            for t in qty:
                if not t["acct"] and entry[1].get(t["norm"]):
                    entry[1][t["norm"]] -= 1
                    t["acct"] = "sentence"
    # pass 3: anything else on the page (another row of a two-table line, a sentence split by the
    # text layer differently), then a sentence that began on the page before
    for i in range(len(lines)):
        for t in per_line[i]:
            if not t["acct"]:
                t["acct"] = nodes.take_from_pool(t["norm"]) or (
                    prev_nodes is not None and i <= 4 and prev_nodes.take_from_pool(t["norm"]) == "sentence"
                    and "sentence") or "residue"
    mark_derivable(per_line, nodes if derive_from is None else derive_from)
    residue = []
    for i, ln in enumerate(lines):
        res = [t for t in per_line[i] if t["acct"] == "residue"]
        if not res:
            continue
        residue.append({"line_no": i, "line": ln.strip()[:240], "class": classify_line(lines, per_line, i, row_of_line),
                        "numbers": [{"text": t["text"], "value": t["value"], "pct": t["pct"], "dollar": t["dollar"],
                                     **({"derivable": t["derived"]} if t.get("derived") else {})} for t in res],
                        "row": row_of_line.get(i, (None,))[0],
                        "above": _neighbour(lines, i, -1), "below": _neighbour(lines, i, 1)})
    return per_line, residue


def _neighbour(lines: list, i: int, step: int) -> str:
    j = i + step
    while 0 <= j < len(lines) and not lines[j].strip():
        j += step
    return lines[j].strip()[:200] if 0 <= j < len(lines) else ""


def _label_only(ln: str, toks: list) -> bool:
    """A short line of words with no quantity: a row label waiting for its figures."""
    words = WORD_RE.findall(ln)
    if not words or any(not t["acct"] == "excluded" for t in toks if t["excl"] != "marker"):
        return False
    s = ln.strip()
    return len(s.split()) <= 12 and not s.endswith((".", ";")) and not s.isupper() or \
        (len(s.split()) <= 6 and not s.endswith("."))


def classify_line(lines: list, per_line: list, i: int, row_of_line: dict) -> str:
    ln = lines[i]
    toks = per_line[i]
    figs, label = tail(ln)
    label_words = WORD_RE.findall(label)
    words = len(ln.split())
    qty = [t for t in toks if t["acct"] != "excluded"]
    if i in row_of_line and any(t["acct"] == "cell" for t in qty):
        return "partial_row"
    values_only = not label_words and len(figs) >= 1
    if values_only:
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j >= 0 and _label_only(lines[j], per_line[j]):
            return "split_row"
        return "orphan"
    if figs and len(label.split()) <= 14 and not re.search(r"[.;]\s*$", ln.strip()) and \
            len(figs) * 3 >= len(qty):
        return "one_line_row"
    if words >= 7:
        return "prose"
    return "orphan"


def load_inputs(raw_path, mentions_path, facts_path):
    rows = json.load(gzip.open(raw_path, "rt", encoding="utf-8"))
    mentions = json.load(gzip.open(mentions_path, "rt", encoding="utf-8"))
    facts = json.loads(Path(facts_path).read_text(encoding="utf-8")) if Path(facts_path).exists() else []
    by_page = defaultdict(lambda: ([], [], []))
    for r in rows:
        by_page[(r["doc"], r["page"])][0].append(r)
    for m in mentions:
        by_page[(m["doc"], m["page"])][1].append(m)
    for f in facts:
        if f.get("page"):
            by_page[(f["doc"], f["page"])][2].append(f)
    return by_page


ACCTS = ("cell", "sentence", "fact", "label", "residue")
EXCLS = ("year", "date", "marker", "page", "reference", "identifier", "enumerator")
CLASSES = ("split_row", "partial_row", "one_line_row", "prose", "orphan")


def run(a):
    by_page = load_inputs(a.raw, a.mentions, a.facts)
    text_root = Path(a.text)
    docs = sorted(d.name for d in text_root.iterdir() if d.is_dir())
    if a.doc:
        docs = [d for d in docs if d == a.doc]
    elif a.ticker:
        docs = [d for d in docs if d.split("_")[0] == a.ticker.upper()]
    per_doc = {}
    all_residue = []
    for doc in docs:
        files = sorted((text_root / doc).glob("p*.txt"))
        pages = [int(f.stem[1:]) for f in files]
        zones = zones_of(files)
        cnt = Counter()
        for f, page in zip(files, pages):
            if a.page and page != a.page:
                continue
            src = page
            if a.control:                                   # another page's nodes: coincidences only
                src = pages[(pages.index(page) + 7) % len(pages)]
            nodes = PageNodes(*by_page.get((doc, src), ([], [], [])))
            prev = PageNodes(*by_page.get((doc, src - 1), ([], [], []))) if not a.control else None
            other = None
            if a.derive_control:                            # shares tested against another page's figures
                other = PageNodes(*by_page.get((doc, pages[(pages.index(page) + 7) % len(pages)]), ([], [], [])))
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            per_line, residue = account_page(lines, nodes, prev, derive_from=other)
            zone = zones[page]
            for toks in per_line:
                for t in toks:
                    cnt["printed"] += 1
                    cnt[t["excl"] or t["acct"]] += 1
                    if not t["excl"]:
                        cnt[f"{zone}_qty"] += 1
                        cnt[f"{zone}_residue"] += t["acct"] == "residue"
                        if t.get("derived"):
                            cnt[f"derivable_{t['derived']}"] += 1
                            cnt[f"{zone}_derivable"] += 1
            for r in residue:
                cnt["residue_lines"] += 1
                cnt[r["class"]] += 1
                if zone == "body":
                    cnt[f"body_{r['class']}"] += 1
                r.update({"doc": doc, "ticker": doc.split("_")[0], "page": page, "zone": zone})
                all_residue.append(r)
            if a.page:
                certificate(doc, page, per_line, residue, zone)
        cnt["pages"] = len(files)
        per_doc[doc] = cnt
    tot = Counter()
    for c in per_doc.values():
        tot.update(c)
    qty = sum(tot[k] for k in ACCTS)
    held = qty - tot["residue"]
    tag = "CONTROL (each page against the nodes of page + 7)" if a.control else "ledger"
    print(f"{tag}: {len(per_doc)} filings, {tot['pages']} pages, {tot['printed']} printed numbers; excluded "
          f"{tot['printed'] - qty} ({', '.join(f'{k} {tot[k]}' for k in EXCLS)}); quantities {qty}: "
          f"in the graph {held} ({held / max(qty, 1):.1%}: cells {tot['cell']}, sentences {tot['sentence']}, "
          f"facts {tot['fact']}, labels {tot['label']}); residue {tot['residue']} numbers in "
          f"{tot['residue_lines']} lines ({', '.join(f'{k} {tot[k]}' for k in CLASSES)})")
    bq, br = tot["body_qty"], tot["body_residue"]
    print(f"by zone: body {bq} quantities, {bq - br} held ({(bq - br) / max(bq, 1):.1%}), residue {br} "
          f"({', '.join(f'{k} {tot[f'body_{k}']}' for k in CLASSES)} lines); tables of contents "
          f"{tot['toc_qty']} (residue {tot['toc_residue']}); exhibits {tot['exhibit_qty']} "
          f"(residue {tot['exhibit_residue']})")
    print(f"residue numbers one division away from two held figures: {tot['derivable_share'] + tot['derivable_change']} "
          f"(share of a figure on the page {tot['derivable_share']}, change between two figures of the line "
          f"{tot['derivable_change']}); in the body {tot['body_derivable']}"
          + ("  [DERIVE CONTROL: shares tested against page + 7]" if a.derive_control else ""))
    if a.page or a.control or a.derive_control:
        return
    out = Path(a.out)
    with gzip.open(out.with_name(out.name + ".tmp"), "wt", encoding="utf-8") as fh:
        json.dump(all_residue, fh)
    out.with_name(out.name + ".tmp").replace(out)
    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["doc", "pages", "printed", "excluded", "quantities", "in_graph", "in_graph_share", "cells",
                    "sentences", "facts", "labels", "residue", "residue_lines"] + list(CLASSES) +
                   ["body_quantities", "body_residue", "body_in_graph_share", "toc_quantities", "exhibit_quantities"])
        for doc, c in sorted(per_doc.items()):
            q = sum(c[k] for k in ACCTS)
            w.writerow([doc, c["pages"], c["printed"], c["printed"] - q, q, q - c["residue"],
                        f"{(q - c['residue']) / max(q, 1):.4f}", c["cell"], c["sentence"], c["fact"], c["label"],
                        c["residue"], c["residue_lines"]] + [c[k] for k in CLASSES] +
                       [c["body_qty"], c["body_residue"], f"{(c['body_qty'] - c['body_residue']) / max(c['body_qty'], 1):.4f}",
                        c["toc_qty"], c["exhibit_qty"]])
    print(f"wrote {out} ({len(all_residue)} residue lines) and {a.csv}")
    print("worst filings by share held:")
    for doc, c in sorted(per_doc.items(), key=lambda kv: (kv[1]["residue"] / max(1, sum(kv[1][k] for k in ACCTS))),
                         reverse=True)[:8]:
        q = sum(c[k] for k in ACCTS)
        print(f"  {doc:22s} {1 - c['residue'] / max(q, 1):.1%} of {q}  residue lines {c['residue_lines']} "
              f"({', '.join(f'{k} {c[k]}' for k in CLASSES)})")


def certificate(doc: str, page: int, per_line: list, residue: list, zone: str = "body"):
    c = Counter()
    for toks in per_line:
        for t in toks:
            c[t["excl"] or t["acct"]] += 1
    printed = sum(c.values())
    qty = sum(c[k] for k in ACCTS)
    print(f"\n{doc} page {page} ({zone}): {printed} numbers printed; {printed - qty} are not quantities "
          f"({', '.join(f'{k} {c[k]}' for k in EXCLS if c[k])}); of {qty} quantities {qty - c['residue']} are in "
          f"the graph (cells {c['cell']}, sentences {c['sentence']}, facts {c['fact']}, labels {c['label']}) and "
          f"{c['residue']} are not, on {len(residue)} lines:")
    for r in residue:
        nums = " ".join(n["text"] for n in r["numbers"])
        print(f"  [{r['class']}] line {r['line_no']}: {r['line'][:110]}")
        print(f"        unread: {nums}" + (f"   (row {r['row']} holds the rest)" if r.get("row") else "") +
              (f"   label above: {r['above'][:60]!r}" if r["class"] == "split_row" else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "graph" / "raw.json.gz"))
    ap.add_argument("--mentions", default=str(ROOT / "graph" / "mentions.json.gz"))
    ap.add_argument("--facts", default=str(ROOT / "data" / "textfacts.json"))
    ap.add_argument("--text", default=str(ROOT.parent / "bench2" / "pdf_text"))
    ap.add_argument("--out", default=str(ROOT / "graph" / "residue.json.gz"))
    ap.add_argument("--csv", default=str(ROOT / "results" / "ledger.csv"))
    ap.add_argument("--ticker", default="")
    ap.add_argument("--doc", default="")
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--derive-control", action="store_true", help="test shares against another page's figures")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
