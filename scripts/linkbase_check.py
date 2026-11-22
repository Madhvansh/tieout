"""Measure the graph's recovered SUM formulas against the SEC's own calculation linkbases.

Every XBRL filing carries a calculation linkbase (*_cal.xml): the parent elements a company
declares and their weighted children. The pipeline never reads it; here, after the build,
it is the independent answer key for the "lines sum to this subtotal" formulas the graph
recovered from arithmetic alone.

    python scripts/linkbase_check.py fetch     # download once, cache under data/sec/linkbase/
    python scripts/linkbase_check.py analyze   # write results/lab_linkbase.json

SEC fair access: User-Agent from SEC_USER_AGENT (.env), at most 5 requests a second,
every file fetched once.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tieout.graph import Graph                                    # noqa: E402
from tieout.llm import load_env                                   # noqa: E402

CACHE = ROOT / "data" / "sec" / "linkbase"
SEC = ROOT / "data" / "sec"
OUT = ROOT / "results" / "lab_linkbase.json"
_last = [0.0]


# ------------------------------------------------------------------ fetching

def _get(url: str) -> bytes:
    import requests
    load_env()
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError("SEC_USER_AGENT is not set")
    for attempt in range(5):                                        # a dropped connection is retried
        wait = 0.25 - (time.time() - _last[0])                      # 4 requests a second at most
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=120)
            _last[0] = time.time()
            r.raise_for_status()
            return r.content
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout):
            _last[0] = time.time()
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def companyfacts(cik) -> dict:
    return json.loads((SEC / f"CIK{int(cik):010d}.json").read_text(encoding="utf-8"))


def find_accession(facts: dict, period_end: str):
    """Accession number of the 10-K whose report date is period_end (companyfacts carries it)."""
    d0 = date.fromisoformat(period_end)
    c = Counter()
    for tax in facts.get("facts", {}).values():
        for body in tax.values():
            for rows in body.get("units", {}).values():
                for x in rows:
                    if x.get("form") == "10-K" and abs((date.fromisoformat(x["end"]) - d0).days) <= 3:
                        c[(x["accn"], x.get("filed"))] += 1
    if not c:
        return None
    # the latest filing of that report date (an amendment would be a 10-K/A, a different form)
    return max(c, key=lambda k: (k[1], c[k]))[0]


def fetch() -> None:
    g = Graph.load()
    CACHE.mkdir(parents=True, exist_ok=True)
    found = {}
    for doc, f in sorted(g.filings.items()):
        if f["role"] != "latest":
            continue
        t = f["ticker"]
        cik = int(g.companies[t]["cik"])
        accn = find_accession(companyfacts(cik), f["period_end"])
        if not accn:
            found[doc] = {"ticker": t, "error": "no 10-K accession in companyfacts"}
            continue
        d = CACHE / f"{t}_{accn}"
        d.mkdir(exist_ok=True)
        idx_path = d / "index.json"
        base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/"
        if not idx_path.exists():
            idx_path.write_bytes(_get(base + "index.json"))
        names = [i["name"] for i in json.loads(idx_path.read_text(encoding="utf-8"))["directory"]["item"]]
        cal = [n for n in names if n.endswith("_cal.xml")]
        rec = {"ticker": t, "accn": accn, "cal": cal[0] if cal else None}
        if cal:
            xsd = cal[0][:-len("_cal.xml")] + ".xsd"
        else:                                                       # linkbases embedded in the schema file
            xs = [n for n in names if n.endswith(".xsd")]
            xsd = xs[0] if xs else None
        for n in ([cal[0]] if cal else []) + ([xsd] if xsd else []):
            p = d / n
            if not p.exists() and n in names:
                p.write_bytes(_get(base + n))
        rec["xsd"] = xsd if xsd in names else None
        found[doc] = rec
        print(doc, rec["accn"], rec["cal"], flush=True)
    (CACHE / "found.json").write_text(json.dumps(found, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ parsing

LINK = "{http://www.xbrl.org/2003/linkbase}"
XLINK = "{http://www.w3.org/1999/xlink}"


def parse_linkbase(d: pathlib.Path, cal, xsd):
    """({role: {parent: {child: weight}}}, {role: definition}) from a filing's cache directory."""
    rel, defs = defaultdict(lambda: defaultdict(dict)), {}
    xroot = ET.parse(d / xsd).getroot() if xsd else None
    if xroot is not None:
        for rt in xroot.iter(LINK + "roleType"):
            df = rt.find(LINK + "definition")
            defs[rt.get("roleURI")] = (df.text or "").strip() if df is not None else ""
    root = ET.parse(d / cal).getroot() if cal else xroot
    for cl in root.iter(LINK + "calculationLink"):
        role = cl.get(XLINK + "role")
        label = {}
        for loc in cl.findall(LINK + "loc"):
            label[loc.get(XLINK + "label")] = loc.get(XLINK + "href").split("#", 1)[-1]
        for arc in cl.findall(LINK + "calculationArc"):
            if arc.get("use") == "prohibited" or not (arc.get(XLINK + "arcrole") or "").endswith("summation-item"):
                continue
            p, c = label.get(arc.get(XLINK + "from")), label.get(arc.get(XLINK + "to"))
            if p and c:
                rel[role][p][c] = float(arc.get("weight", "1"))
    return rel, defs


def role_kind(definition: str) -> str:
    """IS / BS / CF for a primary-statement role, OTHER for equity and comprehensive income, None otherwise."""
    t = definition.lower()
    if "statement" not in t.split(" - ")[1:2] and "- statement -" not in t:
        return None
    if "parenthetical" in t:
        return None
    if "cash flow" in t:
        return "CF"
    if "comprehensive" in t or "equity" in t and "balance" not in t and "financial" not in t or "changes in" in t:
        return "OTHER"
    if any(k in t for k in ("balance sheet", "financial condition", "financial position", "assets and liabilities")):
        return "BS"
    if any(k in t for k in ("operations", "income", "earnings", "profit", "loss")):
        return "IS"
    return "OTHER"


def fact_index(facts: dict, accn: str) -> dict:
    """elem id -> [(start, end, value)] for USD facts the filing itself reports (no dimensions in companyfacts)."""
    out = defaultdict(list)
    for tax, tags in facts.get("facts", {}).items():
        for tag, body in tags.items():
            for x in body.get("units", {}).get("USD", []):
                if x.get("accn") == accn:
                    out[f"{tax}_{tag}"].append((x.get("start"), x["end"], x["val"]))
    return out


def months_of(pid: str):
    if pid.endswith("@end"):
        return None
    if "Q" not in pid:
        return 12
    tail = pid.split("Q", 1)[1]
    q = int(tail[0])
    return 3 * q if tail.endswith("YTD") else 3


def _dm(a: str, b: str) -> float:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days / 30.44


def candidates(r, scale: float, fidx: dict, by_end: dict) -> dict:
    """{element: sign} whose reported value matches the reading at printed precision for its period.
    sign is +1 when the reading and the element carry the same sign, -1 when opposite."""
    out = {}
    if r.unit != "USD" or not r.value:
        return out
    want = months_of(r.period)
    half = 0.5 * scale + 1e-6
    d0 = date.fromisoformat(r.period_end).toordinal()
    for win in (3, 12):                                            # 12: a header that prints only the month (GS)
        for off in range(-win, win + 1):
            for elem, st, en, v in by_end.get(d0 + off, []):
                if abs(abs(v) - abs(r.value)) > half:
                    continue
                if want is not None and st is not None and abs(_dm(st, en) - want) > 0.6:
                    continue
                if want is None and st is not None:
                    continue
                out[elem] = 1 if (v >= 0) == (r.value >= 0) else -1
        if out:
            break
    if want is not None and r.kind == "CF":                         # cash at the start / end of a cash-flow statement
        lo = (date.fromisoformat(r.period_end).toordinal() - round(want * 30.44))
        for off in range(-6, 7):
            for elem, st, en, v in by_end.get(lo + off, []):
                if st is None and abs(abs(v) - abs(r.value)) <= half:
                    out.setdefault(elem, 1 if (v >= 0) == (r.value >= 0) else -1)
        for off in range(-3, 4):
            for elem, st, en, v in by_end.get(d0 + off, []):
                if st is None and abs(abs(v) - abs(r.value)) <= half:
                    out.setdefault(elem, 1 if (v >= 0) == (r.value >= 0) else -1)
    return out


def _words(text: str) -> set:
    w = re.findall(r"[A-Za-z]+", re.sub(r"([a-z])([A-Z])", r" ", text))
    return {x.lower().rstrip("s") for x in w if x.lower() not in ("total", "and", "of", "the", "net", "gaap", "us")}


def _overlap(elem: str, label: str) -> float:
    a, b = _words(elem.split("_", 1)[-1]), _words(label)
    return len(a & b) / len(a | b) if a | b else 0.0


def groups_of(check):
    """[(total reading id, [(reading id, sign)])]: one group per column of the check."""
    out = []
    for rid, m in check.members:
        if m == "total":
            out.append((rid, []))
        else:
            out[-1][1].append((rid, m))
    return out


def rowkey(rid: str) -> str:
    return rid.rsplit(":c", 1)[0]


# ------------------------------------------------------------------ comparison

def expand(formula: dict, rel_role: dict, depth: int = 6) -> dict:
    """Replace every element that is itself a parent in this role by its children, to the leaves."""
    out = defaultdict(float)
    for e, w in formula.items():
        if depth and e in rel_role and len(rel_role[e]) >= 1:
            for e2, w2 in expand(rel_role[e], rel_role, depth - 1).items():
                out[e2] += w * w2
        else:
            out[e] += w
    return {k: v for k, v in out.items() if abs(v) > 1e-9}


def classify(G: dict, L: dict, rel_role: dict) -> str:
    """exact | expanded | subset | superset | conflict | disjoint for graph formula G against relation L."""
    if G == L:
        return "exact"
    shared = set(G) & set(L)
    if not shared:
        eg, el = expand(G, rel_role), expand(L, rel_role)
        return "expanded" if eg == el else "disjoint"
    if any(G[e] != L[e] for e in shared):
        return "conflict"
    if set(G) < set(L):
        return "subset"
    if set(G) > set(L):
        return "superset"
    return "overlap"


RANK = ["exact", "exact_zero_children", "expanded", "subset_ext_only", "ext_swap", "rollforward", "partial_consistent",
        "subset", "superset", "overlap", "conflict", "partial_conflict", "disjoint"]
AGREE = ("exact", "exact_zero_children", "expanded", "subset_ext_only")       # same arithmetic as the declared formula
WEAK = ("subset", "superset")                                                 # graph found a part, or more, of it
DISAGREE = ("overlap", "conflict", "disjoint")
PARTIAL = ("partial_consistent", "partial_conflict")
OWN = ("us-gaap_", "ifrs-full_")


def judge(T: str, G: dict, L: dict, live, rel_role: dict) -> str:
    """Class of graph formula G (element -> effective weight) against one declared relation L.
    live(e): the element carries a non-zero amount in this filing for the period (a zero or absent line adds nothing)."""
    if T in G:                                  # end balance = start balance + change: a roll-forward, not a calc relation
        return "rollforward"
    c = classify(G, L, rel_role)
    if c in ("exact", "expanded"):
        return c
    L2 = {e: w for e, w in L.items() if live(e) or not e.startswith(OWN)}
    if L2 != L:
        c2 = classify(G, L2, rel_role)
        if c2 == "exact":
            return "exact_zero_children"
        c = c2 if RANK.index(c2) < RANK.index(c) else c
        L = L2
    if c == "subset" and all(not e.startswith(OWN) for e in set(L) - set(G)):
        return "subset_ext_only"
    gl, lg = set(G) - set(L), set(L) - set(G)
    if c in ("overlap", "conflict") and gl and lg and len(gl) <= len(lg) and all(not e.startswith(OWN) for e in lg)             and all(G[e] == L[e] for e in set(G) & set(L)):
        return "ext_swap"                       # the SEC element is a company extension; a value-coincident us-gaap element stood in
    return c


def judge_partial(T: str, G: dict, n_unmapped: int, L: dict, live) -> str:
    """Some members of the graph formula could not be mapped to an element (extension line, no SEC amount).
    Consistent when every mapped member is a declared child with the same weight and the unmapped ones fit in the
    declared children the graph did not map."""
    if T in G:
        return "rollforward"
    L2 = {e: w for e, w in L.items() if live(e) or not e.startswith(OWN)}
    if all(e in L2 and L2[e] == w for e, w in G.items()) and n_unmapped <= len(L) - len(G):
        return "partial_consistent"
    return "partial_conflict"


# Disagreements read against the page text (bench2/pdf_text/<doc>/pNNN.txt); every graph sum below adds up on the page.
VERIFIED = [
    {"check": "WFC_FY2025_10K:p91:IS:sum6", "doc": "WFC_FY2025_10K", "page": 91, "class": "overlap",
     "right": "both", "why": "83,699 - 3,658 - 54,842 = 25,199 on the page. The linkbase flattens Total revenue into net interest income + noninterest income; the graph uses the printed subtotal."},
    {"check": "UNH_FY2025_10K:p47:CF:sum4", "doc": "UNH_FY2025_10K", "page": 47, "class": "disjoint",
     "right": "both", "why": "-592 + -355 = -947 on the page. The linkbase declares the net change as operating + investing + financing + FX + cash held for sale; the graph checked the last step of the same reconciliation."},
    {"check": "CAT_FY2025_10K:p66:CF:sum0", "doc": "CAT_FY2025_10K", "page": 66, "class": "ext_swap",
     "right": "graph", "why": "The 14 printed lines add to 11,739. The linkbase has the same 14 children, one a company extension (cat_NonCashGainLossOnDivestiture); the measurement mapped that 30 to a us-gaap element that holds the same amount."},
    {"check": "GM_FY2025_10K:p57:IS:sum1", "doc": "GM_FY2025_10K", "page": 57, "class": "partial_conflict",
     "right": "graph", "why": "159,128 + 14,295 + 8,687 = 182,110. The linkbase lists the same three lines (CostOfGoodsAndServicesSold, OperatingCostsAndExpenses, SG&A); the conflict is measurement noise, two lines unmapped and one value coincidence."},
    {"check": "CVX_FY2025_10K:p66:IS:sum0", "doc": "CVX_FY2025_10K", "page": 66, "class": "partial_conflict",
     "right": "graph", "why": "184,432 + 3,000 + 1,599 = 189,031. The linkbase lists the same three lines, two as company extensions the measurement could not map."},
    {"check": "BLK_FY2025_10K:p80:IS:sum8", "doc": "BLK_FY2025_10K", "page": 80, "class": "partial_conflict",
     "right": "both", "why": "5,942 - 262 - 127 = 5,553. The graph keeps the two printed NCI lines; the linkbase declares one combined NCI element (389)."},
    {"check": "UNH_FY2025_10K:p44:IS:sum0", "doc": "UNH_FY2025_10K", "page": 44, "class": "partial_conflict",
     "right": "graph", "why": "352,229 + 53,380 + 38,038 + 3,920 = 447,567. The linkbase has one aggregate revenue element for products + services (91,418) that is not a printed line."},
    {"check": "WFC_FY2025_10K:p91:IS:sum3", "doc": "WFC_FY2025_10K", "page": 91, "class": "partial_conflict",
     "right": "graph", "why": "The eight printed fees add to 36,215. The linkbase groups three of them under one contract-revenue element that is not printed."},
    {"check": "AMZN_FY2025_10K:p36:CF:sum4", "doc": "AMZN_FY2025_10K", "page": 36, "class": "rollforward",
     "right": "graph", "why": "54,253 + 19,637 = 73,890. End cash = start cash + change is a roll-forward; the linkbase's only relation for that element is cash + restricted cash in a reconciliation note."},
    {"check": "AAPL_FY2026_10K:p37:BS:sum7", "doc": "AAPL_FY2026_10K", "page": 37, "class": "exact_zero_children",
     "right": "both", "why": "292,455 + 112,093 = 404,548. The linkbase also lists Commitments and contingencies, a line with no amount."},
    {"check": "MA_FY2025_10K:p71:BS:sum2", "doc": "MA_FY2025_10K", "page": 71, "class": "partial_conflict",
     "right": "graph", "why": "The seven printed lines add to 22,762; the linkbase has seven children, two of them an extension and a deposit-liability element the measurement mapped to a same-valued asset element."},
]


def analyze() -> None:
    g = Graph.load()
    found = json.loads((CACHE / "found.json").read_text(encoding="utf-8"))
    scale_of = {tid: t["scale"] for tid, t in g.tables.items()}
    sums = defaultdict(list)
    for c in g.checks.values():
        if c.type == "SUM":
            sums[c.doc].append(c)
    rd_by_doc = defaultdict(list)
    for r in g.readings.values():
        rd_by_doc[r.doc].append(r)

    per_company, examples = {}, []
    tot = defaultdict(lambda: defaultdict(Counter))          # kind -> mode -> Counter
    recall = defaultdict(lambda: defaultdict(Counter))       # kind -> mode -> Counter
    mapped_readings = Counter()

    for doc, rec in sorted(found.items()):
        t = rec["ticker"]
        if not rec.get("cal") and not rec.get("xsd"):
            per_company[t] = {"doc": doc, "error": rec.get("error", "no linkbase")}
            continue
        d = CACHE / f"{t}_{rec['accn']}"
        rel, defs = parse_linkbase(d, rec.get("cal"), rec.get("xsd"))
        kind_of_role = {role: role_kind(defs.get(role, "")) for role in rel}
        stm_roles = [r for r in rel if kind_of_role[r]]
        parents_any = defaultdict(list)                           # element -> roles where it is a parent
        for role in rel:
            for p in rel[role]:
                parents_any[p].append(role)
        elems_in_stm = set()
        for role in stm_roles:
            for p, ch in rel[role].items():
                elems_in_stm.add(p)
                elems_in_stm.update(ch)
        facts = companyfacts(g.companies[t]["cik"])
        fidx = fact_index(facts, rec["accn"])
        by_end = defaultdict(list)
        for elem, rows in fidx.items():
            for st, en, v in rows:
                by_end[date.fromisoformat(en).toordinal()].append((elem, st, en, v))
        cand = {}
        for r in rd_by_doc[doc]:
            if r.kind in ("IS", "BS", "CF") and not r.superseded:
                cand[r.id] = candidates(r, scale_of.get(r.table, 1e6), fidx, by_end)
                mapped_readings["USD_readings"] += r.unit == "USD" and bool(r.value)
                mapped_readings["mapped"] += bool(cand[r.id])

        def make_live(ends):
            def live(e):
                return any(v != 0 and en in ends for st, en, v in fidx.get(e, []))
            return live

        def resolve(rows, label=""):
            rows = [x for x in rows if x]                       # a column the SEC does not confirm adds nothing
            if not rows:
                return None, None
            common = set(rows[0])
            for x in rows[1:]:
                common &= set(x)
            if not common:
                return None, None
            if len(common) == 1:
                e = next(iter(common))
                return (e, rows[0][e]), (e, rows[0][e])
            pool = [e for e in common if e in elems_in_stm] or list(common)
            if len(pool) > 1:
                sc = sorted(((_overlap(e, label), e) for e in pool), reverse=True)
                if sc[0][0] > sc[1][0]:
                    pool = [sc[0][1]]
            return (None, (pool[0], rows[0][pool[0]])) if len(pool) == 1 else (None, None)

        comp = defaultdict(Counter)
        formulas = []                                             # (mode, kind, T, G, check id, full?)
        for ck in sums[doc]:
            kind = ck.id.split(":")[2]
            gps = groups_of(ck)
            rowc, lab, rids = defaultdict(list), {}, defaultdict(list)
            for tid, ops in gps:
                for rid in [tid] + [o for o, _ in ops]:
                    rowc[rowkey(rid)].append(cand.get(rid))
                    rids[rowkey(rid)].append(rid)
                    lab[rowkey(rid)] = g.readings[rid].label
            ends = {g.readings[tid].period_end for tid, _ in gps}
            ends = {en for e in ids_end(fidx, ends) for en in [e]}
            live = make_live(ends)
            tk = rowkey(gps[0][0])
            signs = {rowkey(rid): sg for rid, sg in gps[0][1]}
            res = {"check": ck.id, "doc": doc, "kind": kind, "note": ck.note}
            tr = resolve(rowc[tk], lab[tk])
            for mode, idx in (("strict", 0), ("lenient", 1)):
                tot[kind][mode]["checks"] += 1
                comp[kind][f"{mode}:checks"] += 1
                T = tr[idx]
                if T is None:
                    cls = "unmapped_total"
                else:
                    G, unmapped = {}, 0
                    for rk, sg in signs.items():
                        if all(g.readings[x].value == 0 for x in rids[rk]):
                            continue                                # a "-" line adds nothing
                        rr = resolve(rowc[rk], lab[rk])[idx]
                        if rr is None:
                            unmapped += 1
                        else:
                            G[rr[0]] = G.get(rr[0], 0) + sg * rr[1] * T[1]
                    roles = parents_any.get(T[0])
                    if not G:
                        cls = "unmapped_member"
                    elif not roles:
                        cls = "no_relation_ext" if not T[0].startswith(OWN) else "no_relation"
                    else:
                        if unmapped:
                            opts = [(judge_partial(T[0], G, unmapped, rel[r][T[0]], live), r) for r in roles]
                        else:
                            opts = [(judge(T[0], G, rel[r][T[0]], live, rel[r]), r) for r in roles]
                        best = min(opts, key=lambda x: (RANK.index(x[0]), kind_of_role.get(x[1]) is None))
                        cls = best[0]
                        res[mode] = {"total": T[0], "graph": G, "unmapped": unmapped, "role": best[1],
                                     "role_def": defs.get(best[1], ""), "linkbase": rel[best[1]][T[0]], "class": cls}
                        if not unmapped:
                            formulas.append((mode, kind, T[0], G, live))
                    if cls == "no_relation" or cls == "no_relation_ext":
                        res[mode] = {"total": T[0], "graph": G, "unmapped": unmapped, "class": cls}
                    elif cls == "unmapped_member":
                        res[mode] = {"total": T[0], "class": cls}
                tot[kind][mode][cls] += 1
                comp[kind][f"{mode}:{cls}"] += 1
            for mode in ("strict", "lenient"):
                if res.get(mode, {}).get("class") in WEAK + DISAGREE + ("partial_conflict",):
                    examples.append(res)
                    break

        # recall: relations declared on the primary statements against the graph's SUM checks
        tables_kind = defaultdict(set)
        for rid, cs in cand.items():                              # one element per reading, as the lenient mapping does
            if cs:
                _, pick = resolve([cs], g.readings[rid].label)
                if pick:
                    tables_kind[g.readings[rid].kind].add(pick[0])
        seen = set()
        for role in stm_roles:
            k = kind_of_role[role]
            if k not in ("IS", "BS", "CF"):
                continue
            for P, ch in rel[role].items():
                key = (k, P, frozenset(ch.items()))
                if key in seen or len(ch) < 2:
                    continue
                seen.add(key)
                vis = P in tables_kind[k] and all(c in tables_kind[k] for c in ch)
                for mode in ("strict", "lenient"):
                    recall[k][mode]["relations"] += 1
                    comp[k][f"{mode}:lb_relations"] += 1
                    mine = [(G, lv) for m, kk, T, G, lv in formulas if m == mode and kk == k and T == P]
                    agree = any(judge(P, G, ch, lv, rel[role]) in AGREE for G, lv in mine)
                    if P in tables_kind[k]:                      # the graph read the declared total as a line
                        recall[k][mode]["total_read"] += 1
                        recall[k][mode]["total_read_has_sum"] += bool(mine)
                        recall[k][mode]["total_read_agree"] += agree
                    if not vis:
                        continue
                    recall[k][mode]["recoverable"] += 1
                    comp[k][f"{mode}:lb_recoverable"] += 1
                    if mine:
                        recall[k][mode]["touched"] += 1
                        comp[k][f"{mode}:lb_touched"] += 1
                    if agree:
                        recall[k][mode]["recovered_agree"] += 1
                        comp[k][f"{mode}:lb_recovered_agree"] += 1
        per_company[t] = {"doc": doc, "accn": rec["accn"], "statement_roles": len(stm_roles),
                          "declared_parents_on_statements": sum(len(rel[r]) for r in stm_roles),
                          **{k: dict(v) for k, v in comp.items()}}

    summary = {}
    for k, modes in tot.items():
        summary[k] = {}
        for mode, c in modes.items():
            ev = sum(c[x] for x in AGREE + ("ext_swap",) + WEAK + DISAGREE)
            part = c["partial_consistent"] + c["partial_conflict"]
            rc = recall[k][mode]
            summary[k][mode] = {
                "sum_checks": c["checks"],
                "mapped_total_and_all_members": ev + c["rollforward"] + c["no_relation"] + c["no_relation_ext"],
                "compared_with_a_declared_relation": ev,
                "same_arithmetic_as_declared": sum(c[x] for x in AGREE),
                "same_structure_incl_extension_swap": sum(c[x] for x in AGREE) + c["ext_swap"],
                "precision_same_arithmetic": round(sum(c[x] for x in AGREE) / ev, 4) if ev else None,
                "precision_incl_extension_swap": round((sum(c[x] for x in AGREE) + c["ext_swap"]) / ev, 4) if ev else None,
                "no_declared_relation_for_total": c["no_relation"] + c["no_relation_ext"],
                "rollforward_not_a_calc_relation": c["rollforward"],
                "partially_mapped": part, "partially_mapped_consistent": c["partial_consistent"],
                "unmapped_total": c["unmapped_total"], "unmapped_all_members": c["unmapped_member"],
                "recall_declared_relations": rc["relations"],
                "recall_total_read_as_a_line": rc["total_read"],
                "recall_total_read_agree": rc["total_read_agree"],
                "recall_recoverable": rc["recoverable"], "recall_recoverable_agree": rc["recovered_agree"],
                "recall_of_recoverable": round(rc["recovered_agree"] / rc["recoverable"], 4) if rc["recoverable"] else None,
                "recall_of_total_read": round(rc["total_read_agree"] / rc["total_read"], 4) if rc["total_read"] else None,
            }
    out = {"summary": summary, "verified_examples": VERIFIED, "method": "scripts/linkbase_check.py: readings -> XBRL elements by value+period (companyfacts, filing's own "
                     "accession); SUM checks -> effective weights; compared with the filing's cal linkbase",
           "filings": len(found),
           "filings_with_linkbase": sum(1 for r in found.values() if r.get("cal") or r.get("xsd")),
           "separate_cal_files": sum(1 for r in found.values() if r.get("cal")),
           "reading_mapping": dict(mapped_readings),
           "by_kind": {k: {m: dict(c) for m, c in v.items()} for k, v in tot.items()},
           "recall": {k: {m: dict(c) for m, c in v.items()} for k, v in recall.items()},
           "per_company": per_company, "disagreement_candidates": examples}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("filings", "filings_with_linkbase", "separate_cal_files", "reading_mapping",
                                            "summary")}, indent=1))


def ids_end(fidx, ends):
    """The end dates the SEC uses within 12 days of each reading end date."""
    out = set()
    for e in ends:
        d0 = date.fromisoformat(e)
        for rows in fidx.values():
            for st, en, v in rows:
                if abs((date.fromisoformat(en) - d0).days) <= 12:
                    out.add(en)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        fetch()
    elif cmd == "analyze":
        analyze()
    else:
        print(__doc__)
