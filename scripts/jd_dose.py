"""JD direction 1, graph quality against answer accuracy: a dose-response curve on our own graph.

A share p of the graph's numeric nodes is damaged in memory (every layer: statement cells, the other
table rows, sentence figures), the structure reacts as a rebuild would, and the development rounds
are replayed through the full configuration from the cached model calls. Nothing is written to
graph/, semantics/ or cache/. No key is in the environment and every network call is blocked: a model
call the cache does not hold fails, and is counted (`misses`), never made.

    python scripts/jd_dose.py --rate 0.1 --seed 1 --round 7      # one replay -> results/jd_dose_p10s1_r7.json
    python scripts/jd_dose.py --summary                          # every replay -> results/jd_dose.json

Damage, per layer, a share p of the nodes with a non-zero value, each with one fault drawn at random:
a scale fault (x1000 or /1000: a caption read wrong), a sign fault (parentheses lost), a digit fault
(10 percent off). How the structure reacts, by layer:
  statement cells   a SUM or RATIO check with a damaged member fails; a TIE with one is no longer a tie;
                    a CARRY with one reads as a restatement. A cell is then VERIFIED only if another
                    passing check still holds it, FAILED (quarantined: the executor abstains) if it sits
                    in a failing check, else unverified. This is the rule the build applies.
  other table rows  the damaged cell, and every cell of its formula (its total and the total's other
                    parts, in that period column), lose "inside a passing check" (an approximation: a
                    cell held by a second check elsewhere would keep it in a real rebuild).
  sentence figures  the value is damaged; a sentence figure is never verified, so it is FLAGGED anyway.
The text reader's saved replies are not damaged: it reads the filings, not the graph. So the witness
still sees the true figure, and a damaged node it points at is served with the damaged value.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

# the configuration of the reported development runs (dev_scripts/a/run_full.sh with --passage)
CONFIG = {"model": "gemini:gemini-3.5-flash-lite", "checker": "ollama:tieout-planner-4b-v3",
          "arbiter": "gemini:gemini-3.6-flash,gemini:gemini-3.5-flash,gemini:gemini-3-flash-preview",
          "pick_checker": "gemini:gemini-3.1-flash-lite", "passage": "gemini:gemini-3.5-flash-lite"}
ROUNDS = (7, 8, 9, 10)
MISSES: list = []          # (question id, what) for every model call or reader reply the cache did not hold
CURRENT: dict = {}


def guard() -> None:
    """No key in the environment, no network, every model call from the cache (counted when absent)."""
    for k in list(os.environ):
        if "API_KEY" in k or k in ("HF_TOKEN", "OPENROUTER_KEY"):
            del os.environ[k]
    os.environ.update(TIEOUT_LLM=CONFIG["model"], TIEOUT_LINK_LLM=CONFIG["model"],
                      PYTHONIOENCODING="utf-8")
    import requests

    def blocked(*a, **k):
        raise RuntimeError("jd_dose: live call blocked (offline replay)")
    requests.post = requests.get = requests.request = blocked
    requests.Session.request = blocked
    from tieout import llm
    llm.load_env = lambda: None
    real = llm.ask_json

    def counted(*a, **kw):
        try:
            return real(*a, **kw)
        except RuntimeError as e:
            model = kw.get("model") or (a[3] if len(a) > 3 else "")
            MISSES.append({"q": CURRENT.get("qid"), "what": f"model call {model}", "error": str(e)[:80]})
            raise
    llm.ask_json = counted                     # before planner and link import it by name


def fault(v: float, rng: random.Random) -> tuple:
    kind = rng.choice(("scale", "sign", "digit"))
    if kind == "scale":
        return (v * 1000.0 if rng.random() < 0.5 else v / 1000.0), kind
    if kind == "sign":
        return -v, kind
    return v * rng.choice((0.9, 1.1)), kind


def damage(g, store, rate: float, seed: int) -> dict:
    rng = random.Random(seed)
    st: collections.Counter = collections.Counter()
    if rate <= 0:
        return {}
    # statement cells
    ids = sorted(i for i, r in g.readings.items() if isinstance(r.value, (int, float)) and r.value)
    hit = set(rng.sample(ids, round(rate * len(ids))))
    for i in sorted(hit):
        g.readings[i].value, kind = fault(g.readings[i].value, rng)
        st[f"statement cells damaged ({kind})"] += 1
    st["statement cells"] = len(ids)
    touched: set = set()
    for c in g.checks.values():
        if c.status in ("pass", "rounded", "restated") and any(m[0] in hit for m in c.members):
            c.status = {"SUM": "fail", "RATIO": "fail", "TIE": "gone", "CARRY": "restated"}.get(c.type, "fail")
            st[f"checks broken ({c.type})"] += 1
            touched.update(m[0] for m in c.members)
    before = collections.Counter(r.status for r in g.readings.values())
    for i in touched:
        r = g.readings.get(i)
        if r is None:
            continue
        sts = [g.checks[c].status for c in r.checks if c in g.checks]
        r.status = ("verified" if any(s in ("pass", "rounded") for s in sts) else
                    "failed" if any(s == "fail" for s in sts) else "unverified")
    after = collections.Counter(r.status for r in g.readings.values())
    st["statement status before"], st["statement status after"] = dict(before), dict(after)
    # the other table rows
    rows = [r for rs in store.rows_by_ticker.values() for r in rs]
    cells = [(r, k) for r in rows for k, c in enumerate(r["cells"])
             if isinstance(c.get("value"), (int, float)) and c["value"]]
    ok_before = sum(1 for r, k in cells if r["cells"][k].get("ok"))
    for r, k in rng.sample(cells, round(rate * len(cells))):
        c = r["cells"][k]
        c["value"], kind = fault(c["value"], rng)
        st[f"row cells damaged ({kind})"] += 1
        group = [r]
        group += [store.by_id[i] for i in r.get("children") or [] if i in store.by_id]
        parent = store.by_id.get(r.get("parent")) if r.get("parent") else None
        if parent is not None:
            group += [parent] + [store.by_id[i] for i in parent.get("children") or [] if i in store.by_id]
        for row in group:
            for cc in row["cells"]:
                if cc.get("pid") == c.get("pid") and cc.get("ok"):
                    cc["ok"] = False
                    st["row cells that lost a passing check"] += 1
    st["row cells"], st["row cells ok before"] = len(cells), ok_before
    st["row cells ok after"] = sum(1 for r, k in cells if r["cells"][k].get("ok"))
    # sentence figures
    figs = [(m, n) for ms in store.mentions_by_ticker.values() for m in ms for n in m.get("numbers") or []
            if isinstance(n.get("value"), (int, float)) and n["value"]]
    for m, n in rng.sample(figs, round(rate * len(figs))):
        n["value"], kind = fault(n["value"], rng)
        st[f"sentence figures damaged ({kind})"] += 1
    st["sentence figures"] = len(figs)
    return dict(st)


def replay(rate: float, seed: int, rnd: int, transform=None, name: str = "") -> None:
    """Replay one round offline. `transform(g, store) -> stats` replaces the damage (scripts/jd_prune.py)."""
    guard()
    from tieout import graph as graph_mod, link, passage, readers
    real_read = passage.read

    def read(question, model=None, offline=False):
        reply = real_read(question, model=model, offline=offline)
        if reply is None:
            MISSES.append({"q": CURRENT.get("qid"), "what": "text reader reply"})
        return reply
    passage.read = read
    real_answer = readers.answer

    def answer(question, g, panel, **kw):
        CURRENT["qid"] = kw.get("question_id")
        return real_answer(question, g, panel, **kw)
    readers.answer = answer
    stats: dict = {}
    real_load = graph_mod.Graph.load.__func__

    def load(cls, path=graph_mod.GRAPH_PATH):
        g = real_load(cls, path)
        stats.update(transform(g, link.store()) if transform else damage(g, link.store(), rate, seed))
        return g
    graph_mod.Graph.load = classmethod(load)
    import run_benchmark
    tag = f"p{round(rate * 100):02d}s{seed}"
    name = name or f"jd_dose_{tag}_r{rnd}"
    sys.argv = ["run_benchmark.py", "--bench", str(ROOT / "benchmark" / f"round{rnd}" / "dev.jsonl"),
                "--model", CONFIG["model"], "--checker", CONFIG["checker"], "--arbiter", CONFIG["arbiter"],
                "--pick-checker", CONFIG["pick_checker"], "--passage", CONFIG["passage"], "--offline", "--name", name]
    t0 = time.time()
    run_benchmark.main()
    meta = {"rate": rate, "seed": seed, "round": rnd, "seconds": round(time.time() - t0, 1), "damage": stats,
            "misses": MISSES, "questions_with_a_miss": sorted({m["q"] for m in MISSES if m["q"]}),
            "config": CONFIG, "offline": True, "keys_in_environment": 0}
    (RES / f"{name}_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"JD {name}: {meta['seconds']} s, misses {len(MISSES)} on {len(meta['questions_with_a_miss'])} questions")


def summary() -> None:
    by: dict = collections.defaultdict(list)
    for p in sorted(RES.glob("jd_dose_p*_r*_meta.json")):
        meta = json.loads(p.read_text(encoding="utf-8"))
        run = json.loads((RES / p.name.replace("_meta", "")).read_text(encoding="utf-8"))
        by[(meta["rate"], meta["seed"])].append((meta, run))
    ref = {}
    for rnd in ROUNDS:
        f = RES / f"h3_r{rnd}.json"
        if f.exists():
            for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
                ref[(rnd, r["id"])] = bool(r.get("correct"))
    levels = []
    for (rate, seed), runs in sorted(by.items()):
        if sorted(m["round"] for m, _ in runs) != list(ROUNDS):
            continue
        rows = [(m["round"], r) for m, run in runs for r in run["rows"]]
        tiers = collections.defaultdict(lambda: [0, 0])
        for _, r in rows:
            t = r.get("tier") or ("ABSTAIN" if r.get("abstained") else "FLAGGED")
            tiers[t][0] += bool(r.get("correct"))
            tiers[t][1] += 1
        n = len(rows)
        right = sum(1 for _, r in rows if r.get("correct"))
        abstain = sum(1 for _, r in rows if r.get("abstained") and not r.get("correct"))
        miss_q = {(m["round"], q) for m, _ in runs for q in m["questions_with_a_miss"]}
        same = [ref[(k, r["id"])] == bool(r.get("correct")) for k, r in rows if (k, r["id"]) in ref]
        dmg = [m["damage"] for m, _ in runs][0]
        sa = dmg.get("statement status after") or {}
        levels.append({
            "rate": rate, "seed": seed, "n": n, "right": right, "wrong": n - right - abstain, "abstain": abstain,
            "accuracy": round(right / n, 3),
            "plan_road_alone": sum(1 for _, r in rows if (r.get("plan_road") or {}).get("correct")),
            "tiers": {t: f"{v[0]}/{v[1]}" for t, v in sorted(tiers.items())},
            "wrong_while_verified": n and sum(1 for _, r in rows if r.get("tier") == "VERIFIED" and not r.get("correct")),
            "wrong_while_witnessed_or_corroborated": sum(1 for _, r in rows if r.get("tier") in ("WITNESSED", "CORROBORATED")
                                                         and not r.get("correct")),
            "errors": sum(1 for _, r in rows if r.get("error")),
            "questions_with_a_cache_miss": len(miss_q),
            "right_among_questions_without_a_miss": sum(1 for k, r in rows if r.get("correct") and (k, r["id"]) not in miss_q),
            "questions_without_a_miss": sum(1 for k, r in rows if (k, r["id"]) not in miss_q),
            "same_as_reported_run": f"{sum(same)}/{len(same)}",
            "statement_cells_verified_share": round(sa.get("verified", 0) / max(1, sum(sa.values())), 3) if sa else None,
            "row_cells_ok_share": (round(dmg["row cells ok after"] / max(1, dmg["row cells"]), 3) if dmg else None),
            "seconds": round(sum(m["seconds"] for m, _ in runs), 1)})
    out = {"what": "share p of numeric nodes damaged in every layer, rounds 7 to 10 (142 questions) replayed offline "
                   "through the full configuration from cached model calls; see the docstring of scripts/jd_dose.py",
           "command": "python scripts/jd_dose.py --rate P --seed S --round R (each), then --summary",
           "levels": levels}
    (RES / "jd_dose.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for L in levels:
        print(f"p={L['rate']:.2f} s{L['seed']}: right {L['right']}/{L['n']} wrong {L['wrong']} abstain {L['abstain']} | "
              f"tiers {L['tiers']} | wrong while VERIFIED {L['wrong_while_verified']} | misses on "
              f"{L['questions_with_a_cache_miss']} q | same as reported {L['same_as_reported_run']} | "
              f"verified cells {L['statement_cells_verified_share']} ok row cells {L['row_cells_ok_share']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--round", type=int, default=7)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
    else:
        replay(a.rate, a.seed, a.round)


if __name__ == "__main__":
    main()
