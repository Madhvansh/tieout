"""Split the round-2 candidates into the held-out benchmark and a development set.

Three authors, each restricted to the filings and blind to the system, wrote the candidates:
    A  18 single-fact, 9 multi-hop
    B   9 multi-hop, 18 cross-filing
    C  10 single-fact, 5 multi-hop

Author A's questions were read by the developer while the system was still being changed,
so all of A is development material. The held-out benchmark is drawn only from B and C,
whose questions nobody read before the system was frozen:

    single-fact   10  all of C's
    multi-hop     10  C's 5 plus 5 of B's 9, drawn with the seed below
    cross-filing  10  10 of B's 18, drawn with the seed below

Everything else is development. The script prints ids and counts only, never a question.

Usage: python scripts/split_round2.py
"""
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIR = ROOT / "benchmark" / "round2"
SEED = 20261121


def load(name: str) -> list:
    return [json.loads(x) for x in (DIR / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    a, b, c = load("candidates_A.jsonl"), load("candidates_B.jsonl"), load("candidates_C.jsonl")
    rng = random.Random(SEED)
    b_multi = sorted((q for q in b if q["type"] == "multi_hop"), key=lambda q: q["id"])
    b_cross = sorted((q for q in b if q["type"] == "cross_filing"), key=lambda q: q["id"])
    held_ids = {q["id"] for q in rng.sample(b_multi, 5)} | {q["id"] for q in rng.sample(b_cross, 10)}
    held = sorted(c + [q for q in b if q["id"] in held_ids], key=lambda q: (q["type"] != "single_fact", q["type"], q["id"]))
    dev = a + [q for q in b if q["id"] not in held_ids]
    for name, rows in (("questions.jsonl", held), ("dev.jsonl", dev)):
        (DIR / name).write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in rows) + "\n", encoding="utf-8")
    (DIR / "questions_only.jsonl").write_text(
        "\n".join(json.dumps({"id": q["id"], "type": q["type"], "question": q["question"]}, ensure_ascii=False)
                  for q in held) + "\n", encoding="utf-8")
    count = lambda rows: {t: sum(1 for q in rows if q["type"] == t) for t in ("single_fact", "multi_hop", "cross_filing")}
    print("held-out", len(held), count(held), "ids:", " ".join(q["id"] for q in held))
    print("dev     ", len(dev), count(dev))


if __name__ == "__main__":
    main()
