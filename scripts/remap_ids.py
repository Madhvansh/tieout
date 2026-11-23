"""Carry the learned store over a rebuild: every raw-row and sentence id it names is mapped to the id the same
printed line has in the new graph.

A rebuild that finds one more table on a page renumbers every later table of that filing (`DOC:pN:xK:rI`), and
a page read in another order renumbers its sentences (`DOC:pN:sK`). A stored link that kept its old id would
then point at another line, silently. A row is the same row when its filing, page, label and printed cells are
the same; a sentence when its filing, page and text are. An id with no such twin is reported, and the record
that names it is set aside (`state: "lost in rebuild"`), never kept pointing at the wrong line.

    python scripts/remap_ids.py --old-raw OLD --new-raw NEW --old-mentions OLD --new-mentions NEW
                                --store semantics/learned.jsonl --out remapped.jsonl
"""
import argparse
import gzip
import json
import re
from collections import Counter, defaultdict

ID_RE = re.compile(r"[A-Z]{1,5}_FY\d{4}(?:_Q\d)?_10[KQ]:p\d+:(?:x[\w]+:r\d+|s\d+|[A-Z]{1,3}[\w:]*)")


def _load(path):
    return json.load(gzip.open(path, "rt", encoding="utf-8"))


def row_key(r):
    return (r["doc"], r["page"], r["label"], tuple(c.get("text") for c in r["cells"]))


def build_map(old_raw, new_raw, old_m, new_m):
    """old id -> new id (same id when nothing moved), or None when the line has no twin."""
    new_by_key = defaultdict(list)
    for r in new_raw:
        new_by_key[row_key(r)].append(r["id"])
    new_by_label = defaultdict(list)
    for r in new_raw:
        new_by_label[(r["doc"], r["page"], r["label"])].append(r)
    out = {}
    for r in old_raw:
        twins = new_by_key.get(row_key(r), [])
        if r["id"] in twins:
            out[r["id"]] = r["id"]
        elif len(twins) == 1:
            out[r["id"]] = twins[0]
        else:
            # the row gained or lost a cell (a column the fix now reads): same label on the page, and every old
            # printed cell still in the new row, and only one such row
            old_cells = Counter(c.get("text") for c in r["cells"])
            cands = [n for n in new_by_label.get((r["doc"], r["page"], r["label"]), [])
                     if not (old_cells - Counter(c.get("text") for c in n["cells"]))]
            out[r["id"]] = cands[0]["id"] if len(cands) == 1 else None
    new_m_by_key = defaultdict(list)
    new_m_by_page = defaultdict(list)
    for m in new_m:
        new_m_by_key[(m["doc"], m["page"], m["sentence"])].append(m["id"])
        new_m_by_page[(m["doc"], m["page"])].append(m)
    for m in old_m:
        twins = new_m_by_key.get((m["doc"], m["page"], m["sentence"]), [])
        if not twins:
            # a sentence the rebuild completed ("... as of January 31, 2026, and" + "February 1, 2025,
            # respectively."): the only new sentence of the page that starts with the old one, same numbers first
            grown = [n["id"] for n in new_m_by_page.get((m["doc"], m["page"]), [])
                     if n["sentence"].startswith(m["sentence"]) and
                     [x.get("text") for x in n["numbers"][:len(m["numbers"])]] == [x.get("text") for x in m["numbers"]]]
            twins = grown if len(grown) == 1 else []
        out[m["id"]] = m["id"] if m["id"] in twins else (twins[0] if len(twins) == 1 else None)
    return out


def remap_value(v, idmap, seen):
    if isinstance(v, str):
        def sub(mt):
            old = mt.group(0)
            if old not in idmap:
                return old                                  # a statement reading id or anything not rebuilt
            seen[old] = idmap[old]
            return idmap[old] if idmap[old] is not None else old
        return ID_RE.sub(sub, v)
    if isinstance(v, list):
        return [remap_value(x, idmap, seen) for x in v]
    if isinstance(v, dict):
        return {k: remap_value(x, idmap, seen) for k, x in v.items()}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-raw", required=True)
    ap.add_argument("--new-raw", required=True)
    ap.add_argument("--old-mentions", required=True)
    ap.add_argument("--new-mentions", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idmap = build_map(_load(a.old_raw), _load(a.new_raw), _load(a.old_mentions), _load(a.new_mentions))
    moved = sum(1 for k, v in idmap.items() if v is not None and v != k)
    gone = sum(1 for v in idmap.values() if v is None)
    print(f"ids in the old graph {len(idmap)}: same {len(idmap) - moved - gone}, renumbered {moved}, no twin {gone}")
    lines = open(a.store, encoding="utf-8").read().splitlines()
    out, stats = [], Counter()
    for ln in lines:
        if not ln.strip():
            continue
        rec = json.loads(ln)
        seen = {}
        new = remap_value(rec, idmap, seen)
        if any(v is None for v in seen.values()):
            new["state"] = "lost in rebuild"
            new["lost_ids"] = sorted(k for k, v in seen.items() if v is None)
            stats["lost"] += 1
            print(f"  LOST {rec.get('ticker')} {rec.get('phrase') or rec.get('concept')!r}: {new['lost_ids']}")
        elif any(k != v for k, v in seen.items()):
            stats["renumbered"] += 1
        elif seen:
            stats["same"] += 1
        else:
            stats["no raw or sentence id"] += 1
        out.append(json.dumps(new, ensure_ascii=False))
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"store records {len(out)}: {dict(stats)} -> {a.out}")


if __name__ == "__main__":
    main()
