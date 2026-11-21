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
