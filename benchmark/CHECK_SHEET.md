# Benchmark candidates: human check sheet

Open each PDF at the given 1-based page (the number you type into the viewer) and tick each fact.
PDFs: data/pdf/<doc>.pdf

## C01 (single_fact)

Q: What was Apple's top line for fiscal 2026?

Gold: $478,834 million

- [ ] AAPL_FY2026_10K p35: Total net sales (fiscal year ended September 26, 2026) = 478,834 [millions]

Derivation: Printed directly: Total net sales, fiscal 2026 column, in millions.

Notes: Apple fiscal 2026 = 52 weeks ended September 26, 2026. The FY2025 column (416,161) is the trap.

## C02 (single_fact)

Q: How much did JPMorgan Chase earn on the bottom line in 2025?

Gold: $57,048 million

- [ ] JPM_FY2025_10K p167: Net income (year ended December 31, 2025) = 57,048 [millions]

Derivation: Printed directly on the consolidated statement of income.

Notes: Net income applicable to common stockholders is 55,681; the question asks for net income.

## C03 (single_fact)

Q: What net income did Costco report for fiscal 2026?

Gold: $9,226 million

- [ ] COST_FY2026_10K p38: NET INCOME (52 weeks ended August 30, 2026) = 9,226 [millions]

Derivation: Printed directly.

Notes: Costco fiscal 2026 = 52 weeks ended August 30, 2026.

## C04 (single_fact)

Q: How much cash did Cisco generate from operations in fiscal 2026?

Gold: $14,177 million

- [ ] CSCO_FY2026_10K p78: Net cash provided by operating activities (year ended July 25, 2026) = 14,177 [millions]

Derivation: Printed directly on the cash flow statement.

Notes: Cisco fiscal 2026 ended July 25, 2026. Fiscal 2025 (14,193) is nearly identical, so the column matters.

## C05 (single_fact)

Q: What were Oracle's total revenues for fiscal 2026?

Gold: $67,357 million

- [ ] ORCL_FY2026_10K p67: Total revenues (year ended May 31, 2026) = 67,357 [millions]

Derivation: Printed directly.

Notes: Oracle fiscal 2026 ended May 31, 2026.

## C06 (single_fact)

Q: How much in deposits did Goldman Sachs hold at the end of 2025?

Gold: $501,422 million

- [ ] GS_FY2025_10K p142: Deposits (as of December 31, 2025) = 501,422 [millions]

Derivation: Printed directly on the consolidated balance sheet (line includes $76,569 million at fair value).

## C07 (single_fact)

Q: What did Procter & Gamble pay out in dividends in fiscal 2026?

Gold: $10,232 million

- [ ] PG_FY2026_10K p42: Dividends to shareholders (fiscal year ended June 30, 2026) = (10,232) [millions]

Derivation: Cash flow statement financing line, printed as an outflow (10,232); answer is the magnitude.

Notes: P&G fiscal 2026 ended June 30, 2026.

## C08 (single_fact)

Q: How much did Home Depot spend on capital expenditures in fiscal 2025?

Gold: $3,679 million

- [ ] HD_FY2025_10K p51: Capital expenditures (fiscal 2025 (fiscal year ended February 1, 2026)) = (3,679) [millions]

Derivation: Cash flow statement investing line, printed as an outflow; answer is the magnitude.

Notes: Home Depot fiscal 2025 = 52 weeks ended February 1, 2026 (calendar 2026 end date). MD&A rounds it to about $3.7 billion.

## C09 (single_fact)

Q: What was Tesla's total revenue for 2025?

Gold: $94,827 million

- [ ] TSLA_FY2025_10K p53: Total revenues (year ended December 31, 2025) = 94,827 [millions]

Derivation: Printed directly. Total automotive revenues (69,526) is a sub-total, not the answer.

## C10 (single_fact)

Q: How many diluted shares did Apple use to compute fiscal 2026 EPS?

Gold: 14,720,306 thousand shares (about 14.72 billion)

- [ ] AAPL_FY2026_10K p35: Shares used in computing earnings per share: Diluted (fiscal year ended September 26, 2026) = 14,720,306 [thousands]

Derivation: Printed 14,720,306 with the caption saying shares are in thousands, so 14,720,306,000 shares.

Notes: Statement caption: in millions, except number of shares, which are reflected in thousands.

## C11 (single_fact)

Q: In which state is Nike incorporated?

Gold: Oregon

- [ ] NKE_FY2026_10K p2: State or other jurisdiction of incorporation (cover page) = Oregon [text]

Derivation: Cover page: Oregon above the caption State or other jurisdiction of incorporation.

Notes: Also stated in Item 1 on PDF page 4.

## C12 (single_fact)

Q: Which firm audits UnitedHealth Group?

Gold: Deloitte & Touche LLP

- [ ] UNH_FY2025_10K p42: Signature of the report of independent registered public accounting firm (audit of 2025 financial statements) = DELOITTE & TOUCHE LLP [text]

Derivation: Audit report signature, Minneapolis, March 2, 2026.

Notes: PDF page 12 also mentions Deloitte in an executive biography; page 76 mentions KPMG in a director biography (distractor).

## C13 (single_fact)

Q: Roughly how many people did Citigroup employ at the end of 2025?

Gold: About 226,000 (printed as 226 thousand direct staff)

- [ ] C_FY2025_10K p25: Direct staff (in thousands) (at December 31, 2025) = 226 [thousands]

Derivation: Five-year summary prints Direct staff (in thousands) 226, so 226,000.

Notes: Answer '226' without the thousands scale is wrong.

## C14 (single_fact)

Q: How many Class A shares did Facebook's parent have outstanding per the cover of its 2025 annual report?

Gold: 2,187,177,748 Class A shares (as of January 23, 2026)

- [ ] META_FY2025_10K p1: Class A common stock outstanding (cover) (as of January 23, 2026) = 2,187,177,748 [units]

Derivation: Cover page sentence: 2,187,177,748 shares of Class A common stock and 342,377,716 shares of Class B outstanding on January 23, 2026.

Notes: Class B (342,377,716) is the distractor; the count date is after fiscal year-end.

## C15 (single_fact)

Q: What revenue did Microsoft's Intelligent Cloud segment bring in for fiscal 2026?

Gold: $137,791 million

- [ ] MSFT_FY2026_10K p84: Intelligent Cloud Revenue (fiscal year ended June 30, 2026) = 137,791 [millions]

Derivation: Segment note table: Intelligent Cloud, Revenue, 2026 column.

Notes: Total company revenue is 331,839; FY2025 segment revenue is 106,265. Microsoft fiscal 2026 ended June 30, 2026.

## C16 (single_fact)

Q: How many iPhone units did Apple sell in fiscal 2026?

Gold: Not disclosed: the filings in this set report iPhone net sales but not unit volumes.

- [ ] No supporting facts (unanswerable). Confirm the figure is absent from the set.

Derivation: Apple's FY2026 10-K reports net sales by product category but no unit shipments; a search of the filing for unit volumes finds none.

Notes: Correct behaviour is to say the figure is not in the filings; any number is wrong.

## C17 (multi_hop)

Q: What were Apple's net sales in its fourth fiscal quarter of 2026?

Gold: $114,477 million

- [ ] AAPL_FY2026_10K p35: Total net sales (fiscal year ended September 26, 2026) = 478,834 [millions]
- [ ] AAPL_FY2026_Q3_10Q p4: Total net sales (Nine Months Ended) (nine months ended June 27, 2026) = 364,357 [millions]

Derivation: 478,834 (fiscal 2026) - 364,357 (nine months to June 27, 2026) = 114,477 million.

Notes: Fiscal Q4 2026 = the three months ended September 26, 2026 (June 28 to September 26, 2026). The 10-K does not print the quarter.

## C18 (multi_hop)

Q: What revenue did Google's parent book in the fourth quarter of 2025?

Gold: $113,829 million

- [ ] GOOGL_FY2025_10K p50: Revenues (year ended December 31, 2025) = 402,836 [millions]
- [ ] GOOGL_FY2025_Q3_10Q p7: Revenues (Nine Months Ended) (nine months ended September 30, 2025) = 289,007 [millions]

Derivation: 402,836 (full year 2025) - 289,007 (nine months 2025) = 113,829 million.

Notes: Alphabet prints columns oldest-first (2023, 2024, 2025 in the 10-K; 2024 then 2025 in the 10-Q): pick the last column.

## C19 (multi_hop)

Q: How much revenue did Coca-Cola bring in during the fourth quarter of 2025?

Gold: $11,822 million

- [ ] KO_FY2025_10K p63: Net Operating Revenues (year ended December 31, 2025) = 47,941 [millions]
- [ ] KO_FY2025_Q3_10Q p5: Net Operating Revenues (Nine Months Ended) (nine months ended September 26, 2025) = 36,119 [millions]

Derivation: 47,941 (full year 2025) - 36,119 (nine months ended September 26, 2025) = 11,822 million.

Notes: Q4 here = September 27 to December 31, 2025.

## C20 (multi_hop)

Q: What was Microsoft's operating margin in fiscal 2026?

Gold: 46.78%

- [ ] MSFT_FY2026_10K p50: Operating income (fiscal year ended June 30, 2026) = 155,237 [millions]
- [ ] MSFT_FY2026_10K p50: Total revenue (fiscal year ended June 30, 2026) = 331,839 [millions]

Derivation: 155,237 / 331,839 = 0.4678.

Notes: Microsoft fiscal 2026 ended June 30, 2026.

## C21 (multi_hop)

Q: By what percentage did Nvidia grow revenue in fiscal 2026 over fiscal 2025?

Gold: 65.47%

- [ ] NVDA_FY2026_10K p139: Revenue (fiscal year ended January 25, 2026) = 215,938 [millions]
- [ ] NVDA_FY2026_10K p139: Revenue (fiscal year ended January 26, 2025) = 130,497 [millions]

Derivation: 215,938 / 130,497 - 1 = 0.6547.

Notes: Nvidia fiscal 2026 ended January 25, 2026, so most of it is calendar 2025. MD&A may print a rounded percent change.

## C22 (multi_hop)

Q: Using total revenues and its approximate headcount, how much revenue did Walmart generate per associate in fiscal 2026?

Gold: About $339,600 per associate

- [ ] WMT_FY2026_10K p52: Total revenues (fiscal year ended January 31, 2026) = 713,163 [millions]
- [ ] WMT_FY2026_10K p6: Associates (approximate headcount) (as of January 31, 2026) = 2.1 million [text]

Derivation: 713,163 million / 2.1 million associates = $339,601 per associate.

Notes: Walmart fiscal 2026 ended January 31, 2026. Headcount is printed only as approximately 2.1 million (also PDF page 10), so the answer is approximate. Using net sales 706,413 instead gives about $336,387; the question names total revenues.

## C23 (multi_hop)

Q: Who audits the company that owns the Gatorade brand?

Gold: KPMG LLP (PepsiCo's auditor since 1990)

- [ ] PEP_FY2025_10K p4: Brands include Gatorade (business description) = Gatorade [text]
- [ ] PEP_FY2025_10K p113: Signature of the report of independent registered public accounting firm (audit of fiscal 2025 financial statements) = /s/ KPMG LLP [text]

Derivation: Gatorade is a PepsiCo brand (Item 1); PepsiCo's audit report is signed /s/ KPMG LLP.

Notes: Distractor: PDF page 32 says the Controller was once an audit manager at PricewaterhouseCoopers LLP.

## C24 (multi_hop)

Q: What share of its 2025 revenue did Meta spend on research and development?

Gold: 28.55%

- [ ] META_FY2025_10K p90: Research and development (year ended December 31, 2025) = 57,372 [millions]
- [ ] META_FY2025_10K p90: Revenue (year ended December 31, 2025) = 200,966 [millions]

Derivation: 57,372 / 200,966 = 0.2855.

## C25 (multi_hop)

Q: What was Pfizer's R&D spend as a percentage of total revenues in 2025?

Gold: 16.68%

- [ ] PFE_FY2025_10K p58: Research and development expenses (year ended December 31, 2025) = 10,437 [millions]
- [ ] PFE_FY2025_10K p58: Total revenues (year ended December 31, 2025) = 62,579 [millions]

Derivation: 10,437 / 62,579 = 0.1668.

Notes: Excludes the separate Acquired in-process R&D line (1,613). Including it gives 0.1926; the question means the R&D expense line.

## C26 (multi_hop)

Q: What was Caterpillar's operating margin in 2025?

Gold: 16.50%

- [ ] CAT_FY2025_10K p61: Operating profit (year ended December 31, 2025) = 11,151 [millions]
- [ ] CAT_FY2025_10K p61: Total sales and revenues (year ended December 31, 2025) = 67,589 [millions]

Derivation: 11,151 / 67,589 = 0.1650.

## C27 (multi_hop)

Q: Dividing fiscal 2026 membership fee revenue by year-end paid members, what did Costco earn per paid member?

Gold: About $70.24 per paid member

- [ ] COST_FY2026_10K p38: Membership fees (52 weeks ended August 30, 2026) = 5,907 [millions]
- [ ] COST_FY2026_10K p6: Paid members (in thousands) (end of fiscal 2026) = 84,100 [thousands]

Derivation: 5,907 million / 84,100 thousand = 5,907,000,000 / 84,100,000 = $70.24.

Notes: Fees are in millions, members in thousands. Mixing scales gives 0.0702 or 70,238. Costco fiscal 2026 ended August 30, 2026.

## C28 (multi_hop)

Q: How fast did Wells Fargo's net income grow in 2025?

Gold: 8.19%

- [ ] WFC_FY2025_10K p91: Wells Fargo net income (year ended December 31, 2025) = 21,338 [millions]
- [ ] WFC_FY2025_10K p91: Wells Fargo net income (year ended December 31, 2024) = 19,722 [millions]

Derivation: 21,338 / 19,722 - 1 = 0.0819.

Notes: Uses net income attributable to Wells Fargo. Before noncontrolling interests (21,358 vs 19,965) gives 0.0698.

## C29 (multi_hop)

Q: What was Mastercard's net profit margin in 2025?

Gold: 45.65%

- [ ] MA_FY2025_10K p69: Net Income (year ended December 31, 2025) = 14,968 [millions]
- [ ] MA_FY2025_10K p69: Net Revenue (year ended December 31, 2025) = 32,791 [millions]

Derivation: 14,968 / 32,791 = 0.4565.

Notes: Mastercard sits in the banks_finance cluster.

## C30 (multi_hop)

Q: What fraction of its fiscal 2025 earnings did Target hand back as cash dividends?

Gold: 55.41%

- [ ] TGT_FY2025_10K p54: Dividends paid (fiscal 2025 (year ended January 31, 2026)) = (2,053) [millions]
- [ ] TGT_FY2025_10K p51: Net earnings (fiscal 2025 (year ended January 31, 2026)) = 3,705 [millions]

Derivation: 2,053 / 3,705 = 0.5541.

Notes: Target fiscal 2025 ended January 31, 2026. Uses dividends paid (cash flow), not dividends declared.

## C31 (multi_hop)

Q: What share of Amazon's 2025 operating income came from AWS?

Gold: 57.03%

- [ ] AMZN_FY2025_10K p68: AWS Operating income (year ended December 31, 2025) = 45,606 [millions]
- [ ] AMZN_FY2025_10K p68: Consolidated Operating income (year ended December 31, 2025) = 79,975 [millions]

Derivation: 45,606 / 79,975 = 0.5703.

Notes: Amazon prints columns oldest-first (2023, 2024, 2025). Consolidated operating income 79,975 is also on the income statement, PDF page 37.

## C32 (multi_hop)

Q: What was Netflix's operating margin in 2025?

Gold: Not answerable: Netflix is not one of the companies in this filing set.

- [ ] No supporting facts (unanswerable). Confirm the figure is absent from the set.

Derivation: No Netflix filing is in the manifest; Netflix appears only in passing in other companies' filings, with no financials.

## C33 (cross_filing)

Q: Which had the bigger top line in 2025, PepsiCo or Coca-Cola, and by how much?

Gold: PepsiCo, by $45,984 million (93,925 vs 47,941)

- [ ] PEP_FY2025_10K p60: Net Revenue (fiscal year ended December 27, 2025) = 93,925 [millions]
- [ ] KO_FY2025_10K p63: Net Operating Revenues (year ended December 31, 2025) = 47,941 [millions]

Derivation: 93,925 - 47,941 = 45,984 million; PepsiCo is about 1.96x Coca-Cola.

Notes: Gap value 45,984,000,000 USD. PepsiCo's 2025 fiscal year ended December 27, 2025.

## C34 (cross_filing)

Q: Rank JPMorgan, Bank of America, Citigroup and Wells Fargo by deposits at the end of 2025, largest first.

Gold: JPMorgan ($2,559,320M) > Bank of America ($2,018,729M) > Wells Fargo ($1,426,207M) > Citigroup ($1,403,573M)

- [ ] JPM_FY2025_10K p169: Deposits (December 31, 2025) = 2,559,320 [millions]
- [ ] BAC_FY2025_10K p94: Total deposits (December 31, 2025) = 2,018,729 [millions]
- [ ] WFC_FY2025_10K p93: Total deposits (December 31, 2025) = 1,426,207 [millions]
- [ ] C_FY2025_10K p149: Deposits (December 31, 2025) = 1,403,573 [millions]

Derivation: Sort the four year-end deposit balances: 2,559,320 > 2,018,729 > 1,426,207 > 1,403,573.

Notes: Wells Fargo edges Citi by about $22.6 billion; easy to get the last two backwards.

## C35 (cross_filing)

Q: Rank the six big banks in the set (JPMorgan, Bank of America, Citi, Wells Fargo, Goldman Sachs, Morgan Stanley) by 2025 net income attributable to the company.

Gold: JPMorgan (57,048) > Bank of America (30,509) > Wells Fargo (21,338) > Goldman Sachs (17,176) > Morgan Stanley (16,861) > Citigroup (14,306), $ millions

- [ ] JPM_FY2025_10K p167: Net income (2025) = 57,048 [millions]
- [ ] BAC_FY2025_10K p93: Net income (2025) = 30,509 [millions]
- [ ] WFC_FY2025_10K p91: Wells Fargo net income (2025) = 21,338 [millions]
- [ ] GS_FY2025_10K p141: Net earnings (2025) = 17,176 [millions]
- [ ] MS_FY2025_10K p80: Net income applicable to Morgan Stanley (2025) = 16,861 [millions]
- [ ] C_FY2025_10K p146: Citigroup's net income (2025) = 14,306 [millions]

Derivation: Sort the six attributable net income figures.

Notes: Order is the same before noncontrolling interests (MS 17,025, Citi 14,452, WFC 21,358). Goldman vs Morgan Stanley is the close pair (315 apart).

## C36 (cross_filing)

Q: Which carmaker had the larger workforce at the end of 2025, Ford or GM, and by roughly how many people?

Gold: Ford, about 169,000 vs about 155,000 (14,000 more)

- [ ] F_FY2025_10K p21: Total Company employees (in thousands) (December 31, 2025) = 169 [thousands]
- [ ] GM_FY2025_10K p10: Total Worldwide employment (in thousands) (December 31, 2025) = 155 [thousands]

Derivation: Both tables are in thousands: Ford 169 thousand vs GM 155 thousand; 169,000 - 155,000 = 14,000.

Notes: Ford's table prints 2024 first then 2025 (171 then 169): the 2025 figure is the right-hand column. GM's text says about 88,000 hourly plus 68,000 salaried (156,000); the table total is 155 thousand. Ford is larger either way.

## C37 (cross_filing)

Q: Among Merck, J&J and Pfizer, who spent the most on R&D in 2025? Rank all three.

Gold: Merck ($15,789M) > J&J ($14,665M) > Pfizer ($10,437M)

- [ ] MRK_FY2025_10K p79: Research and development (2025) = 15,789 [millions]
- [ ] JNJ_FY2025_10K p50: Research and development expense (fiscal 2025 (ended December 28, 2025)) = 14,665 [millions]
- [ ] PFE_FY2025_10K p58: Research and development expenses (2025) = 10,437 [millions]

Derivation: Sort the three R&D expense lines.

Notes: Order holds if J&J's separate IPR&D impairments (81) and Pfizer's Acquired IPR&D (1,613) are added: J&J 14,746, Pfizer 12,050.

## C38 (cross_filing)

Q: Rank Amazon, Microsoft, Alphabet, Meta and Apple by capital spending in each one's latest fiscal year in the set.

Gold: Amazon ($131,819M) > Microsoft ($115,948M) > Alphabet ($91,447M) > Meta ($69,691M) > Apple ($9,041M)

- [ ] AMZN_FY2025_10K p36: Purchases of property and equipment (year ended December 31, 2025) = (131,819) [millions]
- [ ] MSFT_FY2026_10K p53: Additions to property and equipment (fiscal year ended June 30, 2026) = (115,948) [millions]
- [ ] GOOGL_FY2025_10K p53: Purchases of property and equipment (year ended December 31, 2025) = (91,447) [millions]
- [ ] META_FY2025_10K p93: Purchases of property and equipment (year ended December 31, 2025) = (69,691) [millions]
- [ ] AAPL_FY2026_10K p39: Payments for acquisition of property, plant and equipment (fiscal year ended September 26, 2026) = (9,041) [millions]

Derivation: Sort the five gross property and equipment purchases (cash flow statement).

Notes: Latest fiscal years: Amazon, Alphabet, Meta = calendar 2025; Microsoft = fiscal 2026 ended June 30, 2026; Apple = fiscal 2026 ended September 26, 2026. Amazon net of proceeds (3,499) is 128,320, still first. Alphabet and Amazon print oldest year first.

## C39 (cross_filing)

Q: How much more cash from operations did Exxon generate than Chevron in 2025?

Gold: $18,031 million more (51,970 vs 33,939)

- [ ] XOM_FY2025_10K p76: Net cash provided by operating activities (2025) = 51,970 [millions]
- [ ] CVX_FY2025_10K p69: Net Cash Provided by Operating Activities (2025) = 33,939 [millions]

Derivation: 51,970 - 33,939 = 18,031 million.

## C40 (cross_filing)

Q: Comparing each network's latest annual report in the set, did Visa or Mastercard earn more, and by how much?

Gold: Visa, by $8,536 million (23,504 for fiscal 2026 vs 14,968 for calendar 2025)

- [ ] V_FY2026_10K p60: Net income (fiscal year ended September 30, 2026) = 23,504 [millions]
- [ ] MA_FY2025_10K p69: Net Income (year ended December 31, 2025) = 14,968 [millions]

Derivation: 23,504 - 14,968 = 8,536 million.

Notes: Periods differ: Visa fiscal 2026 ended September 30, 2026; Mastercard fiscal year = calendar 2025. Gap value 8,536,000,000 USD.

## C41 (cross_filing)

Q: Across all 39 companies with a readable latest annual report in the set, which had the smallest balance sheet (total assets) at its latest fiscal year-end?

Gold: Adobe, with total assets of $29,496 million

- [ ] AAPL_FY2026_10K p37: Total assets (September 26, 2026) = 404,548 [millions]
- [ ] MSFT_FY2026_10K p52: Total assets (June 30, 2026) = 758,376 [millions]
- [ ] GOOGL_FY2025_10K p49: Total assets (December 31, 2025) = 595,281 [millions]
- [ ] AMZN_FY2025_10K p39: Total assets (December 31, 2025) = 818,042 [millions]
- [ ] META_FY2025_10K p89: Total assets (December 31, 2025) = 366,021 [millions]
- [ ] NVDA_FY2026_10K p141: Total assets (January 25, 2026) = 206,803 [millions]
- [ ] AMD_FY2025_10K p63: Total assets (December 27, 2025) = 76,926 [millions]
- [ ] ORCL_FY2026_10K p66: Total assets (May 31, 2026) = 261,759 [millions]
- [ ] CRM_FY2026_10K p66: Total assets (January 31, 2026) = 112,305 [millions]
- [ ] ADBE_FY2025_10K p47: Total assets (November 28, 2025) = 29,496 [millions]
- [ ] CSCO_FY2026_10K p75: Total assets (July 25, 2026) = 129,637 [millions]
- [ ] JPM_FY2025_10K p169: Total assets (December 31, 2025) = 4,424,900 [millions]
- [ ] BAC_FY2025_10K p94: Total assets (December 31, 2025) = 3,411,738 [millions]
- [ ] C_FY2025_10K p148: Total assets (December 31, 2025) = 2,657,202 [millions]
- [ ] WFC_FY2025_10K p93: Total assets (December 31, 2025) = 2,148,631 [millions]
- [ ] GS_FY2025_10K p142: Total assets (December 31, 2025) = 1,809,320 [millions]
- [ ] MS_FY2025_10K p81: Total assets (December 31, 2025) = 1,420,270 [millions]
- [ ] V_FY2026_10K p59: Total assets (September 30, 2026) = 94,360 [millions]
- [ ] MA_FY2025_10K p71: Total assets (December 31, 2025) = 54,157 [millions]
- [ ] AXP_FY2025_10K p97: Total assets (December 31, 2025) = 300,052 [millions]
- [ ] BLK_FY2025_10K p79: Total assets (December 31, 2025) = 169,998 [millions]
- [ ] WMT_FY2026_10K p54: Total assets (January 31, 2026) = 284,668 [millions]
- [ ] COST_FY2026_10K p40: Total assets (August 30, 2026) = 89,045 [millions]
- [ ] TGT_FY2025_10K p53: Total assets (January 31, 2026) = 59,490 [millions]
- [ ] KO_FY2025_10K p65: Total assets (December 31, 2025) = 104,816 [millions]
- [ ] PEP_FY2025_10K p64: Total assets (December 27, 2025) = 107,399 [millions]
- [ ] PG_FY2026_10K p40: Total assets (June 30, 2026) = 126,521 [millions]
- [ ] NKE_FY2026_10K p60: Total assets (May 31, 2026) = 38,410 [millions]
- [ ] HD_FY2025_10K p47: Total assets (February 1, 2026) = 105,095 [millions]
- [ ] JNJ_FY2025_10K p49: Total assets (December 28, 2025) = 199,210 [millions]
- [ ] PFE_FY2025_10K p60: Total assets (December 31, 2025) = 208,160 [millions]
- [ ] MRK_FY2025_10K p80: Total assets (unlabeled total line under Other Assets) (December 31, 2025) = 136,866 [millions]
- [ ] UNH_FY2025_10K p43: Total assets (December 31, 2025) = 309,581 [millions]
- [ ] TSLA_FY2025_10K p52: Total assets (December 31, 2025) = 137,806 [millions]
- [ ] F_FY2025_10K p116: Total assets (December 31, 2025) = 289,160 [millions]
- [ ] GM_FY2025_10K p58: Total assets (December 31, 2025) = 281,284 [millions]
- [ ] XOM_FY2025_10K p75: Total assets (December 31, 2025) = 448,980 [millions]
- [ ] CVX_FY2025_10K p68: Total assets (December 31, 2025) = 324,012 [millions]
- [ ] CAT_FY2025_10K p63: Total assets (December 31, 2025) = 98,585 [millions]

Derivation: Collect total assets from all 39 latest balance sheets; the minimum is Adobe 29,496 million, next Nike 38,410, then Mastercard 54,157 and Target 59,490.

Notes: McDonald's is excluded (unreadable text layer). Merck's total assets line is printed without a label ($136,866). Ford prints 2024 first (285,196) then 2025 (289,160). Year-ends differ by company; each uses its latest fiscal year-end.

## C42 (cross_filing)

Q: How many of the 39 readable companies in the set ended their latest fiscal year with more than $1 trillion of total assets, and which are they?

Gold: Six: JPMorgan, Bank of America, Citigroup, Wells Fargo, Goldman Sachs and Morgan Stanley

- [ ] AAPL_FY2026_10K p37: Total assets (September 26, 2026) = 404,548 [millions]
- [ ] MSFT_FY2026_10K p52: Total assets (June 30, 2026) = 758,376 [millions]
- [ ] GOOGL_FY2025_10K p49: Total assets (December 31, 2025) = 595,281 [millions]
- [ ] AMZN_FY2025_10K p39: Total assets (December 31, 2025) = 818,042 [millions]
- [ ] META_FY2025_10K p89: Total assets (December 31, 2025) = 366,021 [millions]
- [ ] NVDA_FY2026_10K p141: Total assets (January 25, 2026) = 206,803 [millions]
- [ ] AMD_FY2025_10K p63: Total assets (December 27, 2025) = 76,926 [millions]
- [ ] ORCL_FY2026_10K p66: Total assets (May 31, 2026) = 261,759 [millions]
- [ ] CRM_FY2026_10K p66: Total assets (January 31, 2026) = 112,305 [millions]
- [ ] ADBE_FY2025_10K p47: Total assets (November 28, 2025) = 29,496 [millions]
- [ ] CSCO_FY2026_10K p75: Total assets (July 25, 2026) = 129,637 [millions]
- [ ] JPM_FY2025_10K p169: Total assets (December 31, 2025) = 4,424,900 [millions]
- [ ] BAC_FY2025_10K p94: Total assets (December 31, 2025) = 3,411,738 [millions]
- [ ] C_FY2025_10K p148: Total assets (December 31, 2025) = 2,657,202 [millions]
- [ ] WFC_FY2025_10K p93: Total assets (December 31, 2025) = 2,148,631 [millions]
- [ ] GS_FY2025_10K p142: Total assets (December 31, 2025) = 1,809,320 [millions]
- [ ] MS_FY2025_10K p81: Total assets (December 31, 2025) = 1,420,270 [millions]
- [ ] V_FY2026_10K p59: Total assets (September 30, 2026) = 94,360 [millions]
- [ ] MA_FY2025_10K p71: Total assets (December 31, 2025) = 54,157 [millions]
- [ ] AXP_FY2025_10K p97: Total assets (December 31, 2025) = 300,052 [millions]
- [ ] BLK_FY2025_10K p79: Total assets (December 31, 2025) = 169,998 [millions]
- [ ] WMT_FY2026_10K p54: Total assets (January 31, 2026) = 284,668 [millions]
- [ ] COST_FY2026_10K p40: Total assets (August 30, 2026) = 89,045 [millions]
- [ ] TGT_FY2025_10K p53: Total assets (January 31, 2026) = 59,490 [millions]
- [ ] KO_FY2025_10K p65: Total assets (December 31, 2025) = 104,816 [millions]
- [ ] PEP_FY2025_10K p64: Total assets (December 27, 2025) = 107,399 [millions]
- [ ] PG_FY2026_10K p40: Total assets (June 30, 2026) = 126,521 [millions]
- [ ] NKE_FY2026_10K p60: Total assets (May 31, 2026) = 38,410 [millions]
- [ ] HD_FY2025_10K p47: Total assets (February 1, 2026) = 105,095 [millions]
- [ ] JNJ_FY2025_10K p49: Total assets (December 28, 2025) = 199,210 [millions]
- [ ] PFE_FY2025_10K p60: Total assets (December 31, 2025) = 208,160 [millions]
- [ ] MRK_FY2025_10K p80: Total assets (unlabeled total line under Other Assets) (December 31, 2025) = 136,866 [millions]
- [ ] UNH_FY2025_10K p43: Total assets (December 31, 2025) = 309,581 [millions]
- [ ] TSLA_FY2025_10K p52: Total assets (December 31, 2025) = 137,806 [millions]
- [ ] F_FY2025_10K p116: Total assets (December 31, 2025) = 289,160 [millions]
- [ ] GM_FY2025_10K p58: Total assets (December 31, 2025) = 281,284 [millions]
- [ ] XOM_FY2025_10K p75: Total assets (December 31, 2025) = 448,980 [millions]
- [ ] CVX_FY2025_10K p68: Total assets (December 31, 2025) = 324,012 [millions]
- [ ] CAT_FY2025_10K p63: Total assets (December 31, 2025) = 98,585 [millions]

Derivation: Count total assets above 1,000,000 million among the 39: JPM 4,424,900; BAC 3,411,738; Citi 2,657,202; WFC 2,148,631; GS 1,809,320; MS 1,420,270. Largest non-bank is Amazon at 818,042.

Notes: Members: JPMorgan Chase, Bank of America, Citigroup, Wells Fargo, Goldman Sachs, Morgan Stanley. McDonald's excluded (unreadable).

## C43 (cross_filing)

Q: Did Apple's fiscal 2026 annual report change the fiscal 2025 net sales figure it had originally reported?

Gold: No. Both filings print fiscal 2025 total net sales of $416,161 million.

- [ ] AAPL_FY2025_10K p32: Total net sales (fiscal year ended September 27, 2025 (as filed in FY2025 10-K)) = 416,161 [millions]
- [ ] AAPL_FY2026_10K p35: Total net sales (fiscal year ended September 27, 2025 (comparative in FY2026 10-K)) = 416,161 [millions]

Derivation: FY2025 10-K current-year column 416,161 = FY2026 10-K prior-year column 416,161; difference 0.

Notes: Not restated. Apple fiscal 2025 = 52 weeks ended September 27, 2025.

## C44 (cross_filing)

Q: What 2024 net income did Bank of America originally report, what does its 2025 annual report now show for 2024, and why the difference?

Gold: Originally $27,132 million; the 2025 10-K shows $26,973 million, $159 million lower, because BofA changed its accounting for tax-related equity investments (proportional amortization) and applied it retrospectively.

- [ ] BAC_FY2024_10K p92: Net income (year ended December 31, 2024 (as filed in FY2024 10-K)) = 27,132 [millions]
- [ ] BAC_FY2025_10K p93: Net income (year ended December 31, 2024 (restated comparative in FY2025 10-K)) = 26,973 [millions]
- [ ] BAC_FY2025_10K p97: Accounting change applied retrospectively (tax-related equity investments) (effective fourth quarter 2025) = applied  retrospectively [text]

Derivation: 26,973 - 27,132 = -159 million.

Notes: Other 2024 lines moved more: total revenue net of interest expense 101,887 -> 105,856 (+3,969); income tax expense 2,122 -> 6,250 (+4,128). answer_value is the change in net income.

## C45 (cross_filing)

Q: Apple's September 2025 balance sheet appears in both of its last two annual reports. Did the 'other non-current assets' figure for that date change, and if so by how much and why?

Gold: Yes. It fell from $83,727 million to $72,634 million (down $11,093 million) because intangible assets, net ($11,093 million) are now shown on their own line; total assets are unchanged at $359,241 million.

- [ ] AAPL_FY2025_10K p34: Other non-current assets (September 27, 2025 (as filed in FY2025 10-K)) = 83,727 [millions]
- [ ] AAPL_FY2026_10K p37: Other non-current assets (September 27, 2025 (comparative in FY2026 10-K)) = 72,634 [millions]
- [ ] AAPL_FY2026_10K p37: Intangible assets, net (September 27, 2025 (comparative in FY2026 10-K)) = 11,093 [millions]
- [ ] AAPL_FY2026_10K p40: Reclassification note (Note 1) = reclassified [text]

Derivation: 72,634 - 83,727 = -11,093 = minus the new Intangible assets, net line (11,093). 72,634 + 11,093 = 83,727.

Notes: A reclassification, not a change in totals. Note 1 says certain prior period amounts were reclassified to conform to current presentation.

## C46 (cross_filing)

Q: Did Coca-Cola's latest 10-K change the year-end 2024 balance for prepaid expenses and other current assets versus what it first reported?

Gold: Yes. It went from $3,129 million to $2,998 million (down $131 million); the $131 million now appears as a separate Assets held for sale line. Total current assets stay $25,997 million.

- [ ] KO_FY2024_10K p66: Prepaid expenses and other current assets (December 31, 2024 (as filed in FY2024 10-K)) = 3,129 [millions]
- [ ] KO_FY2025_10K p65: Prepaid expenses and other current assets (December 31, 2024 (comparative in FY2025 10-K)) = 2,998 [millions]
- [ ] KO_FY2025_10K p65: Assets held for sale (December 31, 2024 (comparative in FY2025 10-K)) = 131 [millions]

Derivation: 2,998 - 3,129 = -131 = minus Assets held for sale (131) in the 2024 comparative column.

Notes: Same pattern on the liability side: Accounts payable and accrued expenses 21,715 -> 21,712 with 3 shown as Liabilities held for sale. KO notes prior-year amounts were revised to conform to current presentation (PDF page 68).

## C47 (cross_filing)

Q: In Microsoft's fiscal 2025 cash flow statement, did the depreciation, amortization, and other add-back change between the fiscal 2025 and fiscal 2026 annual reports?

Gold: Yes. It was $34,153 million as first filed and $29,433 million in the fiscal 2026 10-K, a cut of $4,720 million, offset by a matching rise in net recognized losses on investments and derivatives (609 to 5,329); cash from operations is unchanged at $136,162 million.

- [ ] MSFT_FY2025_10K p53: Depreciation, amortization, and other (fiscal year ended June 30, 2025 (as filed in FY2025 10-K)) = 34,153 [millions]
- [ ] MSFT_FY2026_10K p53: Depreciation, amortization, and other (fiscal year ended June 30, 2025 (comparative in FY2026 10-K)) = 29,433 [millions]
- [ ] MSFT_FY2025_10K p53: Net recognized losses on investments and derivatives (fiscal 2025 as filed) = 609 [millions]
- [ ] MSFT_FY2026_10K p53: Net recognized losses (gains) on investments and derivatives (fiscal 2025 comparative) = 5,329 [millions]
- [ ] MSFT_FY2026_10K p53: Net cash from operations (fiscal 2025 comparative) = 136,162 [millions]

Derivation: 29,433 - 34,153 = -4,720; 5,329 - 609 = +4,720; net cash from operations 136,162 in both.

Notes: Reclassification within operating activities; Note 1 of the FY2026 10-K (PDF page 55) says prior period cash flow amounts were recast. Microsoft fiscal 2025 ended June 30, 2025.

## C48 (cross_filing)

Q: How did Intel's 2025 revenue compare with AMD's?

Gold: Not answerable: Intel is not in this filing set. AMD's 2025 net revenue was $34,639 million, but no Intel figures are available.

- [ ] AMD_FY2025_10K p61: Net revenue (year ended December 27, 2025) = 34,639 [millions]

Derivation: Only one side of the comparison exists in the set. AMD's 10-K names Intel as a competitor but gives no Intel financials.

Notes: A partial answer giving AMD's figure is fine; any Intel number is a hallucination.
