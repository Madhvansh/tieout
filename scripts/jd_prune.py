"""JD direction 2, pruning and consolidation, measured on our graph.

In this graph a node that a recovered formula recomputes from other nodes is redundant: it can be
dropped and solved for when asked. This script counts such nodes exactly, checks that each one
recomputes to its printed value, measures what dropping them saves, and replays the development
rounds with two pruned graphs:

  solvable   concept facts the shipped executor already solves when they are missing (the identities
             in semantics/metrics.yaml, and a fourth quarter as the year minus nine months), dropped
             only where the solved value equals the printed one. Drop and let the executor solve.
  rowcells   cells of the other table rows that their row's recovered formula recomputes exactly
             (one per equation and period column), dropped with NO solver: what answers lose when a
             graph is pruned without solving.

Also counted: consolidation (sentence figures that restate a table cell; one fact read from two
filings) and conflicts (figures two filings print differently, and what the graph says about each,
tieout/restate.py).

    python scripts/jd_prune.py --analyze                      # counts, exactness, sizes -> results/jd_prune.json
    python scripts/jd_prune.py --replay solvable --round 7    # one replay (offline, no key) -> results/jd_prune_solvable_r7.json
    python scripts/jd_prune.py --summary                      # adds the replays to results/jd_prune.json
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

COL = re.compile(r":c(\d+)$")
ROUNDS = (7, 8, 9, 10)


def _exact(total: float, parts: list) -> bool:
    s = sum(sign * v for sign, v in parts)
    return abs(total - s) <= 1e-9 * max(1.0, abs(total))


def greedy(equations: list) -> tuple:
    """equations: [(member ids, preferred id to drop)]. One node per equation, never two unknowns in an
    equation that recomputes a dropped node. Returns (dropped set, {dropped: equation index})."""
    dropped, locked, used_by = set(), set(), {}
    for k, (members, prefer) in enumerate(equations):
        if any(m in dropped for m in members):
            continue
        order = [prefer] + [m for m in members if m != prefer]
        pick = next((m for m in order if m not in locked), None)
        if pick is None:
            continue
        dropped.add(pick)
        used_by[pick] = k
        locked.update(m for m in members if m != pick)
    return dropped, used_by


def statement_equations(g) -> tuple:
    """Per SUM check and period column: (members, the total) for exact equations; also how many were not exact."""
    eqs, not_exact = [], 0
    for c in g.checks.values():
        if c.type != "SUM" or c.status not in ("pass", "rounded"):
            continue
        by_col = collections.defaultdict(list)
        for rid, role in c.members:
            m = COL.search(rid)
            by_col[m.group(1) if m else ""].append((rid, role))
        for col, ms in by_col.items():
            tot = [rid for rid, role in ms if role == "total"]
            parts = [(role, rid) for rid, role in ms if role != "total"]
            if len(tot) != 1 or not parts or any(rid not in g.readings for rid, _ in ms):
                continue
            t = g.readings[tot[0]].value
            if not _exact(t, [(float(s), g.readings[r].value) for s, r in parts]):
                not_exact += 1
                continue
            eqs.append(([tot[0]] + [r for _, r in parts], tot[0]))
    return eqs, not_exact


def row_equations(store) -> tuple:
    """Per formula row of the other tables and period: (cell ids "<row id>#<cell index>", the total's
    cell id) for the exact ones. A row's cell for a period is the one the executor reads
    (link.period_cells), so a dropped cell is exactly the figure a linked line would serve."""
    from tieout.link import period_cells
    eqs, not_exact = [], 0
    for rows in store.rows_by_ticker.values():
        for r in rows:
            kids = [store.by_id.get(i) for i in r.get("children") or []]
            signs = r.get("signs") or [1] * len(kids)
            if len(kids) < 2 or any(k is None for k in kids) or len(signs) != len(kids):
                continue
            kid_cells = [period_cells(k) for k in kids]
            for pid, j in period_cells(r).items():
                c = r["cells"][j]
                if not c.get("ok") or not isinstance(c.get("value"), (int, float)):
                    continue
                parts = []
                for k, s, pc in zip(kids, signs, kid_cells):
                    kc = k["cells"][pc[pid]] if pid in pc else None
                    if kc is None or not kc.get("ok") or not isinstance(kc.get("value"), (int, float)):
                        parts = None
                        break
                    parts.append((s, kc["value"], f"{k['id']}#{pc[pid]}"))
                if not parts:
                    continue
                if not _exact(c["value"], [(s, v) for s, v, _ in parts]):
                    not_exact += 1
                    continue
                eqs.append(([f"{r['id']}#{j}"] + [x for _, _, x in parts], f"{r['id']}#{j}"))
    return eqs, not_exact


def solvable_facts(g) -> list:
    """Concept facts the executor's _derive recomputes exactly from facts that stay."""
    from tieout.concepts import concept_defs
    from tieout.execute import OUTFLOW
    from tieout.plan import identity_defs
    concepts = concept_defs()

    def val(t, c, p):
        f = g.fact(t, c, p)
        if f is None:
            return None
        r = g.best_reading(f)
        return None if r is None else (abs(r.value) if c in OUTFLOW else r.value)
    drop, keep = [], set()
    keys = sorted(g.facts)
    for key in keys:
        t, c, p = key.split("|")
        for ident in identity_defs():                 # the executor takes the first identity it can fill
            if c not in ident or key in keep:
                continue
            others = [(n, val(t, n, p)) for n in ident if n != c]
            if any(v is None for _, v in others) or any(f"{t}|{n}|{p}" in drop for n, _ in others):
                continue
            solved = -sum(ident[n] * v for n, v in others) / ident[c]
            solved = abs(solved) if c in OUTFLOW else solved      # as the executor serves it
            mine = val(t, c, p)
            if mine is not None and abs(solved - mine) <= 1e-9 * max(1.0, abs(mine)):
                drop.append(key)
                keep.update(f"{t}|{n}|{p}" for n, _ in others)
            break                                     # not exact: the executor would serve another figure
        if key in drop or key in keep:
            continue
        m = re.fullmatch(r"FY(\d{4})Q4", p)
        if m and concepts.get(c) is not None and concepts[c].nature == "flow":
            y, ytd = val(t, c, f"FY{m.group(1)}"), val(t, c, f"FY{m.group(1)}Q3YTD")
            mine = val(t, c, p)
            if None not in (y, ytd, mine) and abs((y - ytd) - mine) <= 1e-9 * max(1.0, abs(mine)):
                drop.append(key)
                keep.update({f"{t}|{c}|FY{m.group(1)}", f"{t}|{c}|FY{m.group(1)}Q3YTD"})
    return drop


def _gz(obj) -> int:
    return len(gzip.compress(json.dumps(obj).encode("utf-8"), 6))


def analyze() -> dict:
    from dataclasses import asdict
    from tieout import link
    from tieout.graph import Graph
    from tieout.restate import events
    g = Graph.load()
    st = link.store()
    out: dict = {"what": "pruning and consolidation measured on the shipped graph; see the docstring of scripts/jd_prune.py",
                 "command": "python scripts/jd_prune.py --analyze"}
    # statement cells recomputable by a recovered formula
    s_eqs, s_not = statement_equations(g)
    s_drop, _ = greedy(s_eqs)
    out["statement_cells"] = {"cells": len(g.readings), "exact_formula_equations": len(s_eqs),
                              "equations_not_exact (rounded)": s_not, "droppable_one_per_equation": len(s_drop),
                              "share": round(len(s_drop) / len(g.readings), 4)}
    # duplicates: a value printed twice on one filing (TIE) or last year's figure in this year's filing (CARRY)
    dup = collections.Counter()
    for c in g.checks.values():
        if c.status == "pass" and c.type in ("TIE", "CARRY"):
            dup[c.type] += 1
    out["duplicate_readings"] = {"TIE pairs (printed twice in one filing)": dup["TIE"],
                                 "CARRY pairs (one figure in two filings, equal)": dup["CARRY"],
                                 "facts held by readings of two or more filings": sum(
                                     1 for f in g.facts.values() if len({g.readings[i].doc for i in f.readings if i in g.readings}) >= 2)}
    # cells of the other tables recomputable by their row's formula
    r_eqs, r_not = row_equations(st)
    r_drop, _ = greedy(r_eqs)
    rows = [r for rs in st.rows_by_ticker.values() for r in rs]
    ncells = sum(len(r["cells"]) for r in rows)
    out["row_cells"] = {"rows": len(rows), "cells": ncells, "exact_formula_equations": len(r_eqs),
                        "equations_not_exact (rounded)": r_not, "droppable_one_per_equation": len(r_drop),
                        "share": round(len(r_drop) / ncells, 4)}
    # sizes: the serialised layers with and without the droppable nodes (gzip, as shipped)
    readings = [asdict(r) for r in g.readings.values()]
    out["sizes_bytes_gzip"] = {
        "statement_readings_all": _gz(readings),
        "statement_readings_without_droppable": _gz([r for r in readings if r["id"] not in s_drop])}
    pruned_rows = []
    for r in rows:
        keep = [c for j, c in enumerate(r["cells"]) if f"{r['id']}#{j}" not in r_drop]
        pruned_rows.append({**r, "cells": keep})
    out["sizes_bytes_gzip"]["row_layer_all"] = _gz(rows)
    out["sizes_bytes_gzip"]["row_layer_without_droppable"] = _gz(pruned_rows)
    # what the shipped executor already solves for (drop and solve)
    sf = solvable_facts(g)
    out["solvable_facts"] = {"facts": len(g.facts), "droppable": len(sf),
                             "by_concept": dict(collections.Counter(k.split("|")[1] for k in sf))}
    # consolidation: sentence figures that restate a table cell
    ms = [m for xs in st.mentions_by_ticker.values() for m in xs]
    tied = [m for m in ms if m.get("ties")]
    nfig = sum(len([n for n in m.get("numbers") or [] if isinstance(n.get("value"), (int, float))]) for m in ms)
    out["consolidation"] = {"sentences": len(ms), "sentence_figures": nfig,
                            "sentences_whose_figure_restates_a_table_cell": len(tied),
                            "table_cells_restated_by_a_sentence": len({c for m in tied for c in m["ties"]})}
    # conflicts: figures two filings print differently, and the graph's reading of each
    ev = events(g, explain=True)
    out["conflicts"] = {"events": len(ev), "by_kind": dict(collections.Counter(e["kind"] for e in ev)),
                        "carry_checks_restated": sum(1 for c in g.checks.values() if c.type == "CARRY" and c.status == "restated"),
                        "examples": [{k: e.get(k) for k in ("ticker", "statement", "period", "kind", "old_doc", "new_doc")}
                                     | {"line": (e["line"].get("label") if isinstance(e.get("line"), dict) else str(e.get("line")))}
                                     for e in ev[:6]]}
    old = json.loads((RES / "jd_prune.json").read_text(encoding="utf-8")) if (RES / "jd_prune.json").exists() else {}
    if old.get("replays"):
        out["replays"] = old["replays"]
    (RES / "jd_prune.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("conflicts",)}, indent=1, default=str)[:3000])
    print("conflicts", out["conflicts"]["events"], out["conflicts"]["by_kind"])
    return out


def transform_solvable(g, store) -> dict:
    drop = solvable_facts(g)
    for k in drop:
        del g.facts[k]
    return {"facts dropped": len(drop), "by_concept": dict(collections.Counter(k.split("|")[1] for k in drop))}


def transform_rowcells(g, store) -> dict:
    eqs, _ = row_equations(store)
    drop, _ = greedy(eqs)
    n = 0
    for rows in store.rows_by_ticker.values():
        for r in rows:
            before = len(r["cells"])
            r["cells"] = [c for j, c in enumerate(r["cells"]) if f"{r['id']}#{j}" not in drop]
            n += before - len(r["cells"])
    return {"row cells dropped": n}


def summary() -> None:
    out = json.loads((RES / "jd_prune.json").read_text(encoding="utf-8")) if (RES / "jd_prune.json").exists() else {}
    ref = {}
    for rnd in ROUNDS:
        f = RES / f"jd_dose_p00s1_r{rnd}.json"
        if f.exists():
            ref[rnd] = json.loads(f.read_text(encoding="utf-8"))["rows"]
    reps = {}
    for mode in ("solvable", "rowcells"):
        runs = []
        for rnd in ROUNDS:
            f, m = RES / f"jd_prune_{mode}_r{rnd}.json", RES / f"jd_prune_{mode}_r{rnd}_meta.json"
            if f.exists() and m.exists():
                runs.append((rnd, json.loads(f.read_text(encoding="utf-8")), json.loads(m.read_text(encoding="utf-8"))))
        if len(runs) != len(ROUNDS):
            continue
        rows = [(k, r) for k, run, _ in runs for r in run["rows"]]
        base = [(k, r) for k in ROUNDS for r in ref.get(k, [])]
        miss_q = {(k, q) for k, _, meta in runs for q in meta["questions_with_a_miss"]}
        b_ok = {(k, r["id"]): bool(r.get("correct")) for k, r in base}
        flips = [(k, r["id"]) for k, r in rows if (k, r["id"]) in b_ok and b_ok[(k, r["id"])] != bool(r.get("correct"))]
        reps[mode] = {"right": sum(1 for _, r in rows if r.get("correct")), "n": len(rows),
                      "unpruned_right": sum(1 for _, r in base if r.get("correct")), "unpruned_n": len(base),
                      "answers_changed": len(flips), "changed": [f"r{k} {q}" for k, q in flips][:20],
                      "questions_with_a_cache_miss": len(miss_q),
                      "seconds_per_answer": round(sum(r.get("seconds", 0) for _, r in rows) / max(1, len(rows)), 2),
                      "unpruned_seconds_per_answer": round(sum(r.get("seconds", 0) for _, r in base) / max(1, len(base)), 2),
                      "dropped": runs[0][2]["damage"]}
    out["replays"] = reps
    (RES / "jd_prune.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(reps, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--replay", choices=["solvable", "rowcells"])
    ap.add_argument("--round", type=int, default=7)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    elif a.replay:
        import jd_dose
        jd_dose.replay(0.0, 0, a.round, transform={"solvable": transform_solvable, "rowcells": transform_rowcells}[a.replay],
                       name=f"jd_prune_{a.replay}_r{a.round}")
    elif a.summary:
        summary()


if __name__ == "__main__":
    main()
