"""Offline prompt-size measurement for the LLM budget plan. No LLM call.

    python -m baseline.budget [--out baseline/budget_measure.json]

For every development question (the three dev files, de-duplicated by id, unanswerable included) it
builds the exact prompt the baseline would send and reports characters and estimated tokens.
Tokens are an ESTIMATE: characters / 4 (the usual English rule) and characters / 3 (a conservative
figure for number-heavy filing tables). Real counts come back from the provider on live calls and
are recorded per question by scripts/run_baseline.py (tokens_in, tokens_out).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline import answer, hybrid, index          # noqa: E402

DEV_FILES = ["benchmark/round2/dev.jsonl", "benchmark/questions.jsonl", "benchmark/dev.jsonl"]


def questions() -> list:
    seen, out = set(), []
    for f in DEV_FILES:
        for line in (ROOT / f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                q = json.loads(line)
                if q["id"] not in seen:
                    seen.add(q["id"])
                    out.append(q)
    return out


def dist(xs: list) -> dict:
    xs = sorted(xs)
    return {"mean": round(statistics.mean(xs)), "p50": round(xs[len(xs) // 2]),
            "p95": round(xs[int(0.95 * (len(xs) - 1))]), "max": round(xs[-1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "baseline" / "budget_measure.json"))
    args = ap.parse_args()
    qs = questions()
    result = {"questions": len(qs)}
    for name, kw in [("v2", {"retriever": "v2"}), ("v1", {"retriever": "v1"})]:
        (hybrid if name == "v2" else index).load()
        chars, passages = [], []
        for q in qs:
            b = answer.build_prompt(q["question"], **kw)
            chars.append(len(b["system"]) + len(b["user"]))
            passages.append(len(b["retrieved"]))
        result[name] = {"prompt_chars": dist(chars), "est_tokens_chars_div_4": dist([c / 4 for c in chars]),
                        "est_tokens_chars_div_3": dist([c / 3 for c in chars]),
                        "items_per_prompt": dist(passages), "system_prompt_chars": len(
                            answer.PASSAGE_SYSTEM if name == "v2" else answer.SYSTEM)}
        print(name, json.dumps(result[name]), flush=True)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
