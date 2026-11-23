# Vision lane report (B item 3; lane done 00:02 IST Mon 23 Nov 2026; saved by Session B from the lane's final message)

Question: can a page image read by a vision model recover table lines the rule-based build cannot read, with every
recovered number accepted only by arithmetic? The model may arrange printed numbers; it may never invent one.

Setup: `gemini-3.5-flash-lite` (free tier model, development key), 13 pages: MS FY2025 p143, JPM FY2025 p318 and p320,
then the 10 pages with the most ledger residue (JPM p263, NVDA p71, JNJ x4, BAC x2, C p198, CRM p51). 13 calls, mean
47.9 s per page. `python vision.py --offline` reproduces identical output hashes from `vision_lab/cache/`.

Accepted:
- Cells proposed 2,508; every one passed the text-layer check (its printed text is on the page), so arithmetic is the
  real filter.
- Rows proposed 445, accepted 318 (by a footing equation 308, by a tie to an existing raw row 47, tie only 10);
  arranged but not proven 110 (not accepted).
- Footing equations proven: 79 totals (nested subtotals, signed "Other = Total minus parts", totals above or below their
  parts, whole-table sums), tolerance one unit of the last printed digit, chance of a lucky match bounded at 1e-4 with
  decoy totals. A tie needs 2 cells to match one raw row with the same label and years.
- Of 1,906 cells in accepted rows, 1,110 sit in an equation's columns; rate cells are text-layer only (12 of 12 also
  match interest over balance).

Residue recovered on these pages: 935 of 1,337 numbers (262 of 387 lines). JPM p318 134/140, JPM p320 156/168,
C p198 82/82; MS p143 20 of 22 rows accepted (its ledger residue was 1 number).

Control: numbers shuffled within each page, 100 seeds: 0 accepted rows in every seed. The first, looser acceptance
accepted a mean of 3.0 rows (max 9) and was tightened (chance bound, 2-cell ties, 2+ operands per column, signed
footings limited to 4 operands).

Labelled comparison, `gemini-3.1-pro-preview` (paid key): JPM p320 27 accepted rows and JNJ p30 31, the same as
flash-lite; MS p143 36 of 42 rows (flash-lite 20 of 22). 3 answers over 12 requests (the rest 429s).

Limits: labels and the row-to-label assignment are not proven (244 of 245 labels have all their words on the page);
columns without a footing are unchecked; roll-forwards and sparse tables (JPM p263, NVDA p71, parts of the JNJ 10-Q)
recover little because they do not foot; only 13 pages; the rows are NOT wired into the graph.

Files (`takehome/vision_lab/`): `vision.py`, `vision_rows.json.gz` (accepted), `arranged_unproven.json.gz`,
`metrics.json`, `cache/` (16 responses), `vision_rows_pro.json.gz`, `arranged_unproven_pro.json.gz`, `metrics_pro.json`,
`inspect_page.py`, `control_detail.py`. Status log: `train_lab/STATUS_VISION.md`.
