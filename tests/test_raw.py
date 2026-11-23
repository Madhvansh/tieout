"""Raw layer tests that need no PDF: synthetic tables and word lines."""
from tieout.alltables import _side_by_side
from tieout.rawlayer import _Profile, row_unit, table_rows
from tieout.tables import Cell, Column, Row, Table, Word


def make_table(rows, caption="", title="", headers=None, scale=1e6, scale_source="inherited", dollar_rows=()):
    ncols = max(len(v) for _, v in rows)
    headers = headers or [""] * ncols
    t = Table(doc="T", page=3, title=title, caption=caption, scale=scale, share_scale=scale,
              columns=[Column(i, 300.0 + 60 * i, 270.0 + 60 * i, header=headers[i]) for i in range(ncols)], rows=[])
    t.scale_source, t.intro, t.context = scale_source, "", ""
    for i, (label, vals) in enumerate(rows):
        r = Row(idx=i, page=3, y=10.0 * i, label=label, indent=0.0, label_bbox=None)
        for c, v in enumerate(vals):
            if v is None:
                continue
            text = v if isinstance(v, str) else f"{v:,}"
            num = float(text.replace(",", "").replace("%", "").strip("()")) * (-1 if text.startswith("(") else 1)
            r.cells[c] = Cell(page=3, bbox=(0, 0, 1, 1), text=text, value=num, col=c,
                              is_percent=text.endswith("%"), has_dollar=label in dollar_rows)
        t.rows.append(r)
    return t


def rows_of(t):
    return table_rows("T", "TT", 0, t, [(None, None)] * t.ncols, {})


def test_store_counts_are_not_money():
    t = make_table([("Alabama", [23, "3,153"]), ("California", [321, "38,067"]), ("Total", ["1,995", "250,518"])],
                   caption="Stores Retail Square Feet (in thousands)", headers=["Stores", "Retail Square Feet (in thousands)"])
    out = rows_of(t)
    assert all(r["unit"] == "count" for r in out)
    ca = out[1]["cells"]
    assert ca[0]["value"] == 321                     # stores: units, never the money scale
    assert ca[1]["value"] == 38_067_000              # square feet: the column's own "(in thousands)"


def test_dollar_sign_keeps_money():
    t = make_table([("Stores segment", [100, 200]), ("Total", [300, 400])], dollar_rows=("Stores segment", "Total"))
    out = rows_of(t)
    assert out[0]["unit"] == "USD" and out[0]["cells"][0]["value"] == 100_000_000


def test_count_row_in_money_table_takes_its_label_scale():
    t = make_table([("Total deposits", ["1,403,573", "1,284,458"]), ("Direct staff (in thousands)", [226, 229])],
                   caption="In millions of dollars", scale_source="caption")
    out = rows_of(t)
    assert out[0]["unit"] == "USD" and out[0]["cells"][0]["value"] == 1_403_573_000_000
    assert out[1]["unit"] == "count" and out[1]["cells"][0]["value"] == 226_000


def test_billions_table_scales_decimals():
    t = make_table([("Total client assets", ["7.4", "6.9"]), ("Net new assets", ["0.3", "0.2"])],
                   caption="($ in billions)", scale=1e9, scale_source="caption")
    out = rows_of(t)
    assert out[0]["unit"] == "USD" and out[0]["cells"][0]["value"] == 7_400_000_000


def test_exception_clause_scales_one_row():
    t = make_table([("Unearned revenue", ["24,317", "20,743"]), ("Remaining performance obligation", ["72.4", "63.4"])],
                   caption="(in millions, except remaining performance obligation, which is presented in billions)",
                   scale=1e9, scale_source="caption", dollar_rows=("Unearned revenue",))
    out = rows_of(t)
    assert out[0]["cells"][0]["value"] == 24_317_000_000
    assert out[1]["unit"] == "USD" and out[1]["cells"][0]["value"] == 72_400_000_000


def test_percent_and_per_share_rows():
    t = make_table([("Revenue", [100, 90]), ("Growth", ["11%", "4%"]), ("Diluted earnings per share", ["6.62", "6.02"])],
                   caption="(in millions, except per share)", scale_source="caption", dollar_rows=("Revenue",))
    units = [r["unit"] for r in rows_of(t)]
    assert units == ["USD", "percent", "USD/share"]


def test_percent_cell_in_money_row_is_not_scaled():
    t = make_table([("Subscription", ["39,388", "35,679", "10%"]), ("Services", ["2,137", "2,216", "(4)%"])],
                   dollar_rows=("Subscription",))
    cell = rows_of(t)[0]["cells"][2]
    assert cell["unit"] == "percent" and cell["value"] == 10


def test_parent_children_and_ids():
    t = make_table([("Products", ["294,866", "298,085"]), ("Services", ["96,169", "85,200"]),
                    ("Total net sales", ["391,035", "383,285"])], dollar_rows=("Total net sales",))
    out = rows_of(t)
    total = out[2]
    assert total["role"] == "total" and total["children"] == [out[0]["id"], out[1]["id"]]
    assert out[0]["parent"] == total["id"] and total["parent"] is None
    assert all(r["table_id"] == "T:p3:x0" and r["id"].startswith("T:p3:x0:r") for r in out)
    assert all("ci" in c and "col" in c for r in out for c in r["cells"])


def test_undated_columns_keep_their_cells():
    t = make_table([("Owned", ["1,546", 41]), ("Leased", [288, 27])], caption="Stores Facilities",
                   headers=["Stores", "Facilities"])
    out = rows_of(t)
    assert [c["col"] for c in out[0]["cells"]] == ["Stores", "Facilities"]
    assert out[0]["cells"][0]["pid"] is None and out[0]["cells"][0]["value"] == 1546


def test_profile_reads_column_scale():
    t = make_table([("Alabama", [23, "3,153"]), ("Alaska", [3, 504])],
                   headers=["Stores", "(in thousands)"], caption="Stores (in thousands)")
    prof = _Profile(t)
    assert prof.count and prof.col_scale == {1: 1e3} and not prof.own_scale
    assert row_unit(t, t.rows[0], prof) == "count"


def _line(y, *parts):
    """parts: (x0, text) pairs; words 6 points wide per character."""
    return [Word(x, y, x + 6 * len(s), y + 8, s) for x, s in parts]


def test_side_by_side_tables_are_split():
    names = [("Alabama", "Montana"), ("Alaska", "Nebraska"), ("Arizona", "Nevada"), ("Arkansas", "Ohio"),
             ("California", "Texas")]
    stream = [_line(20 + 12 * i, (50, a), (200, "12"), (260, "1,000"), (330, b), (470, "7"), (540, "777"))
              for i, (a, b) in enumerate(names)]
    parts = _side_by_side(stream)
    assert len(parts) == 2
    assert [ln[0].text for ln in parts[0]] == [a for a, _ in names]
    assert [ln[0].text for ln in parts[1]] == [b for _, b in names]


def test_ordinary_table_is_not_split():
    stream = [_line(20 + 12 * i, (50, lab), (300, "1,000"), (380, "2,000")) for i, lab in
              enumerate(["Revenue", "Cost of sales", "Gross margin", "Operating expenses", "Operating income"])]
    assert len(_side_by_side(stream)) == 1
