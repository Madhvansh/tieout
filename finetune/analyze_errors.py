"""Classify planner mistakes in a val_predictions.jsonl (from train_kaggle.py) or a run_local.py dump.

Usage: python analyze_errors.py out_local_full/val_predictions.jsonl [--top 10]
Each wrong row gets one or more labels: invalid_json, cannot_mismatch, step_count, op_sequence,
or field:<key> for the first-differing keys of aligned steps (company, concept, period, where ...), answer_id.
"""
import collections
import json
import sys


def labels(gold, pred_txt):
    try:
        pred = json.loads(pred_txt)
    except Exception:
        return ["invalid_json"]
    if not isinstance(pred, dict):
        return ["invalid_json"]
    if ("cannot" in gold) != ("cannot" in pred):
        return ["cannot_mismatch"]
    gs, ps = gold.get("steps", []), pred.get("steps", [])
    if len(gs) != len(ps):
        return ["step_count"]
    if [s.get("op") for s in gs] != [s.get("op") for s in ps]:
        return ["op_sequence"]
    out = []
    for g, p in zip(gs, ps):
        for k in sorted(set(g) | set(p)):
            if g.get(k) != p.get(k):
                if k in ("concept", "metric", "item") and (k not in g or k not in p):
                    out.append("field:concept_vs_metric_vs_item")
                else:
                    out.append(f"field:{k}")
    if gold.get("answer") != pred.get("answer"):
        out.append("answer_id")
    if gold.get("cannot") != pred.get("cannot") and "cannot" in gold:
        out.append("cannot_text")
    return sorted(set(out)) or ["other"]


def main():
    path = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 10
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    c = collections.Counter(); ex = {}; wrong = 0
    for r in rows:
        gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
        try:
            ok = json.loads(r["pred"]) == gold
        except Exception:
            ok = False
        if ok:
            continue
        wrong += 1
        for lb in labels(gold, r["pred"]):
            c[lb] += 1
            ex.setdefault(lb, r)
    print(f"{len(rows)} rows, {wrong} wrong ({wrong / len(rows):.1%})")
    for lb, n in c.most_common(top):
        e = ex[lb]
        print(f"{n:4d}  {lb}\n      Q: {e['question'][:110]}\n      gold: {str(e['gold'])[:160]}\n      pred: {str(e['pred'])[:160]}")


if __name__ == "__main__":
    main()
