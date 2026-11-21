"""Split the 48 frozen candidates into the 30-question benchmark and an 18-question development set.

The split is random, stratified by question type (10 per type to the benchmark, 6 per type
to development), with a fixed seed. Development questions are the only ones used while
debugging the system; the benchmark questions are run once the system is frozen.

Usage: python scripts/split_benchmark.py   (writes benchmark/questions.jsonl and benchmark/dev.jsonl)
"""
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 20261121
TYPES = ("single_fact", "multi_hop", "cross_filing")


def main():
    cands = [json.loads(x) for x in (ROOT / "benchmark" / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rng = random.Random(SEED)
    test, dev = [], []
    for t in TYPES:
        group = sorted((c for c in cands if c["type"] == t), key=lambda c: c["id"])
        rng.shuffle(group)
        test.extend(group[:10])
        dev.extend(group[10:])
    for name, rows in (("questions", test), ("dev", dev)):
        rows.sort(key=lambda c: c["id"])
        with open(ROOT / "benchmark" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("benchmark:", [c["id"] for c in test])
    print("development:", [c["id"] for c in dev])
    for name, rows in (("benchmark", test), ("development", dev)):
        tags = {}
        for r in rows:
            for tag in r.get("tags", []):
                tags[tag] = tags.get(tag, 0) + 1
        print(name, "tags:", dict(sorted(tags.items())))


if __name__ == "__main__":
    main()
