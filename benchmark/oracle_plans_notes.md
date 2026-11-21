# Oracle plan notes (blind; written from the questions and plan_language.txt only)

- C08: Home Depot's fiscal year ends in late January/early February; used {"fy": 2025} as the company labels it (rule 6), not the calendar year.
- C14: language falls short. Per-class share counts are not attributes; shares_outstanding may return the total or Class A plus Class B, not Class A alone. attr also takes no period, so "per the cover of its 2025 annual report" cannot be pinned.
- C15: segment revenue via "item"; label chosen as "Intelligent Cloud revenue" (the filing table may print just "Intelligent Cloud" under a revenue heading).
- C17: Apple's fourth fiscal quarter is a period the 10-K may not print as its own column (Q4 = FY minus 9-month YTD); plan uses {"fy": 2026, "q": 4} as the language intends.
- C19: same Q4 caveat as C17 for Coca-Cola's 2025 fourth quarter.
- C24, C25: used rnd_intensity because the question asks exactly "R&D as a share/percentage of revenue"; a get-plus-calc plan would be equivalent.
- C27: unsure. "Paid members" is a count Costco prints in an MD&A table, so it is planned as an "item" rather than "cannot"; it is a year-end balance, so fy 2026 is taken at the end of the fiscal year. Membership fee item label "Membership fees" follows the income statement.
- C31: AWS segment operating income via "item" ("AWS operating income"); share computed by calc over company OperatingIncome.
- C32: cannot; Netflix is not in the company list.
- C35: listed the six tickers explicitly instead of the banks_finance cluster, which likely also holds AXP, BLK, V, MA. NetIncome is already "attributable to the company".
- C36: language falls short. rank does not support attributes, so the plan gets both employee counts and returns F minus GM; the sign says which is larger. attr takes no period, so "end of 2025" relies on the latest filing.
- C40: "earn more" read as NetIncome; "each network's latest annual report" read as {"fy": "latest"} per company (fiscal years differ: Visa ends September, Mastercard December).
- C42: language falls short. The answer names one step, so it is the count (s2); the companies themselves ("which are they") sit in filter step s1. "39 readable companies" taken as "all".
- C44: plan covers the number only (restated = latest minus first reported for 2024); the "why" is not a fact the language returns, and the two individual reported values are not separately addressable.
- C45: Apple's September 2025 balance sheet = its fiscal 2025 year end, so {"fy": 2025}; restated compares the first report (FY2025 10-K) with the latest (FY2026 10-K). The "why" part is not expressible.
- C46: restated on item "Prepaid expenses and other current assets" at {"fy": 2024}.
- C48: cannot; Intel is not in the company list (only AMD is), so no partial comparison was planned.
