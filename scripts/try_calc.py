"""Dev helper: parse given pages of one PDF and print the recovered formulas."""
import sys

from tieout.calc import coverage, discover, explain
from tieout.tables import page_table


def main():
    pdf = sys.argv[1]
    pages = [int(x) for x in sys.argv[2:]]
    for p in pages:
        t = page_table(pdf, p, pdf)
        if t is None:
            print(p, "no table")
            continue
        checks = discover(t)
        hit, total = coverage(t, checks)
        print(f"== page {p} {t.title[:50]} | rows {len(t.rows)} | formulas {len(checks)} "
              f"| verified cells {hit}/{total}")
        print(explain(t, checks))
        ok_rows = {r for ch in checks if ch.status != 'fail' for r in ch.rows}
        loose = [r.label[:40] for r in t.rows if r.idx not in ok_rows]
        print("  not in any formula:", loose)


if __name__ == "__main__":
    main()
