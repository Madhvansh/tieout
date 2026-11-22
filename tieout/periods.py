"""Fiscal calendars.

"FY2025" is not a date range until you know the company: Apple's ends in
September 2025, Walmart's in January 2025, Target's in January 2026. Each
company gets a calendar from its own latest annual report, and every printed
period end is mapped to a period id on that calendar:

    FY2025          the twelve months of fiscal 2025
    FY2025@end      the balance-sheet date closing fiscal 2025
    FY2026Q3        the third fiscal quarter (three months)
    FY2026Q3YTD     fiscal year to date at the end of the third quarter
    FY2026Q3@end    the balance-sheet date closing that quarter

Period ids, not raw dates, are what filings are joined on, so the prior-year
column of one filing meets the current-year column of the previous one even
for 52/53-week filers whose year end moves by a few days.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class FiscalCalendar:
    ticker: str
    fye_month: int
    fye_day: int
    offset: int          # fiscal label minus the calendar year in which that fiscal year ends

    @classmethod
    def from_latest(cls, ticker: str, fiscal_year: str, period_end: str) -> "FiscalCalendar":
        end = date.fromisoformat(period_end)
        return cls(ticker, end.month, end.day, int(fiscal_year.lstrip("FY")) - end.year)

    def fye_in(self, year: int) -> date:
        day = min(self.fye_day, calendar.monthrange(year, self.fye_month)[1])
        return date(year, self.fye_month, day)

    def end_of_label(self, label_year: int) -> date:
        """Approximate year-end date of the fiscal year printed as `label_year`."""
        return self.fye_in(label_year - self.offset)

    def period_id(self, end: date, months: Optional[int], instant: bool) -> str:
        nearest = min((self.fye_in(y) for y in (end.year - 1, end.year, end.year + 1)),
                      key=lambda f: abs((f - end).days))
        if abs((nearest - end).days) <= 20:
            fy = nearest.year + self.offset
            if instant:
                return f"FY{fy}@end"
            if months in (None, 12):
                return f"FY{fy}"
            return f"FY{fy}Q4" if months == 3 else f"FY{fy}M{months}"
        nxt = min(f for f in (self.fye_in(y) for y in (end.year, end.year + 1)) if f > end)
        prev = self.fye_in(nxt.year - 1)
        q = max(1, min(3, round((end - prev).days / 91.3)))
        fy = nxt.year + self.offset
        if instant:
            return f"FY{fy}Q{q}@end"
        if months in (None, 3):
            return f"FY{fy}Q{q}"
        if months == 3 * q:
            return f"FY{fy}Q{q}YTD"
        return f"FY{fy}Q{q}M{months}"


FUTURE_SLACK_DAYS = 14      # a 52/53-week year end estimated from another year's day can run a few days late


def _same_day(year: int, like: date) -> date:
    return date(year, like.month, min(like.day, calendar.monthrange(year, like.month)[1]))


def column_period(cal: FiscalCalendar, col, form: str, instant: bool = False,
                  filing_end: Optional[str] = None, heading: str = "") -> tuple:
    """(period id, ISO end date) for one table column, or (None, None). One function for the
    primary statements (graph) and every other table (rawlayer).

    A column with a printed date ends on that date. A column that prints only a year:
      in a 10-K, the fiscal year so labelled ends on the company's year-end day;
      in a 10-Q, the column is a quarter or a year to date, never a fiscal year. It ends on the
        filing's own period end (month and day) in that year, and its length is what the column
        or else the table's title or heading says ("three", "six", "nine months"). With no length
        printed, or for a balance-sheet column of another year, it gets no period id.
    No column gets a period that ends after the filing's own period end: a filing reports nothing
    that has not ended.
    """
    from .tables import _part_year                  # tables does not import periods: no cycle
    fe = date.fromisoformat(filing_end) if filing_end else None
    months = None if instant else col.months
    if col.end_date:
        try:
            end = date.fromisoformat(col.end_date)
        except ValueError:
            return None, None
    elif col.year and 1995 <= col.year <= 2035:
        if form == "10-Q" and fe is not None:
            end = _same_day(col.year, fe)
            if instant:
                if col.year != fe.year:
                    return None, None
            else:
                months = months if months and months < 12 else _part_year(heading or "")
                if not months:
                    return None, None
        else:
            end = cal.end_of_label(col.year)          # header printed only a fiscal-year label
    else:
        return None, None
    if fe is not None and (end - fe).days > FUTURE_SLACK_DAYS:
        return None, None
    if not instant and months is None and form == "10-K":
        months = 12
    return cal.period_id(end, months, instant), end.isoformat()
