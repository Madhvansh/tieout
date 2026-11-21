# Gold check sheet: the 30 benchmark questions

For each question open the PDF at the page given and confirm the printed value. Tick or correct.
PDFs are in `data/pdf/<document>.pdf`. Page numbers are PDF page numbers (what the viewer shows), not printed ones.

## C01 (single_fact)

Question: What was Apple's top line for fiscal 2026?

Gold answer: $478,834 million

- [ ] `AAPL_FY2026_10K.pdf` page 35: "Total net sales" = 478,834 (fiscal year ended September 26, 2026, in millions)

Derivation: Printed directly: Total net sales, fiscal 2026 column, in millions.

Note: Apple fiscal 2026 = 52 weeks ended September 26, 2026. The FY2025 column (416,161) is the trap.

## C03 (single_fact)

Question: What net income did Costco report for fiscal 2026?

Gold answer: $9,226 million

- [ ] `COST_FY2026_10K.pdf` page 38: "NET INCOME" = 9,226 (52 weeks ended August 30, 2026, in millions)

Derivation: Printed directly.

Note: Costco fiscal 2026 = 52 weeks ended August 30, 2026.

## C04 (single_fact)

Question: How much cash did Cisco generate from operations in fiscal 2026?

Gold answer: $14,177 million

- [ ] `CSCO_FY2026_10K.pdf` page 78: "Net cash provided by operating activities" = 14,177 (year ended July 25, 2026, in millions)

Derivation: Printed directly on the cash flow statement.

Note: Cisco fiscal 2026 ended July 25, 2026. Fiscal 2025 (14,193) is nearly identical, so the column matters.

## C07 (single_fact)

Question: What did Procter & Gamble pay out in dividends in fiscal 2026?

Gold answer: $10,232 million

- [ ] `PG_FY2026_10K.pdf` page 42: "Dividends to shareholders" = (10,232) (fiscal year ended June 30, 2026, in millions)

Derivation: Cash flow statement financing line, printed as an outflow (10,232); answer is the magnitude.

Note: P&G fiscal 2026 ended June 30, 2026.

## C08 (single_fact)

Question: How much did Home Depot spend on capital expenditures in fiscal 2025?

Gold answer: $3,679 million

- [ ] `HD_FY2025_10K.pdf` page 51: "Capital expenditures" = (3,679) (fiscal 2025 (fiscal year ended February 1, 2026), in millions)

Derivation: Cash flow statement investing line, printed as an outflow; answer is the magnitude.

Note: Home Depot fiscal 2025 = 52 weeks ended February 1, 2026 (calendar 2026 end date). MD&A rounds it to about $3.7 billion.

## C09 (single_fact)

Question: What was Tesla's total revenue for 2025?

Gold answer: $94,827 million

- [ ] `TSLA_FY2025_10K.pdf` page 53: "Total revenues" = 94,827 (year ended December 31, 2025, in millions)

Derivation: Printed directly. Total automotive revenues (69,526) is a sub-total, not the answer.

## C11 (single_fact)

Question: In which state is Nike incorporated?

Gold answer: Oregon

- [ ] `NKE_FY2026_10K.pdf` page 2: "State or other jurisdiction of incorporation" = Oregon (cover page, in text)

Derivation: Cover page: Oregon above the caption State or other jurisdiction of incorporation.

Note: Also stated in Item 1 on PDF page 4.

## C12 (single_fact)

Question: Which firm audits UnitedHealth Group?

Gold answer: Deloitte & Touche LLP

- [ ] `UNH_FY2025_10K.pdf` page 42: "Signature of the report of independent registered public accounting firm" = DELOITTE & TOUCHE LLP (audit of 2025 financial statements, in text)

Derivation: Audit report signature, Minneapolis, March 2, 2026.

Note: PDF page 12 also mentions Deloitte in an executive biography; page 76 mentions KPMG in a director biography (distractor).

## C14 (single_fact)

Question: How many Class A shares did Facebook's parent have outstanding per the cover of its 2025 annual report?

Gold answer: 2,187,177,748 Class A shares (as of January 23, 2026)

- [ ] `META_FY2025_10K.pdf` page 1: "Class A common stock outstanding (cover)" = 2,187,177,748 (as of January 23, 2026, in units)

Derivation: Cover page sentence: 2,187,177,748 shares of Class A common stock and 342,377,716 shares of Class B outstanding on January 23, 2026.

Note: Class B (342,377,716) is the distractor; the count date is after fiscal year-end.

## C15 (single_fact)

Question: What revenue did Microsoft's Intelligent Cloud segment bring in for fiscal 2026?

Gold answer: $137,791 million

- [ ] `MSFT_FY2026_10K.pdf` page 84: "Intelligent Cloud Revenue" = 137,791 (fiscal year ended June 30, 2026, in millions)

Derivation: Segment note table: Intelligent Cloud, Revenue, 2026 column.

Note: Total company revenue is 331,839; FY2025 segment revenue is 106,265. Microsoft fiscal 2026 ended June 30, 2026.

## C17 (multi_hop)

Question: What were Apple's net sales in its fourth fiscal quarter of 2026?

Gold answer: $114,477 million

- [ ] `AAPL_FY2026_10K.pdf` page 35: "Total net sales" = 478,834 (fiscal year ended September 26, 2026, in millions)
- [ ] `AAPL_FY2026_Q3_10Q.pdf` page 4: "Total net sales (Nine Months Ended)" = 364,357 (nine months ended June 27, 2026, in millions)

Derivation: 478,834 (fiscal 2026) - 364,357 (nine months to June 27, 2026) = 114,477 million.

Note: Fiscal Q4 2026 = the three months ended September 26, 2026 (June 28 to September 26, 2026). The 10-K does not print the quarter.

## C19 (multi_hop)

Question: How much revenue did Coca-Cola bring in during the fourth quarter of 2025?

Gold answer: $11,822 million

- [ ] `KO_FY2025_10K.pdf` page 63: "Net Operating Revenues" = 47,941 (year ended December 31, 2025, in millions)
- [ ] `KO_FY2025_Q3_10Q.pdf` page 5: "Net Operating Revenues (Nine Months Ended)" = 36,119 (nine months ended September 26, 2025, in millions)

Derivation: 47,941 (full year 2025) - 36,119 (nine months ended September 26, 2025) = 11,822 million.

Note: Q4 here = September 27 to December 31, 2025.

## C20 (multi_hop)

Question: What was Microsoft's operating margin in fiscal 2026?

Gold answer: 46.78%

- [ ] `MSFT_FY2026_10K.pdf` page 50: "Operating income" = 155,237 (fiscal year ended June 30, 2026, in millions)
- [ ] `MSFT_FY2026_10K.pdf` page 50: "Total revenue" = 331,839 (fiscal year ended June 30, 2026, in millions)

Derivation: 155,237 / 331,839 = 0.4678.

Note: Microsoft fiscal 2026 ended June 30, 2026.

## C21 (multi_hop)

Question: By what percentage did Nvidia grow revenue in fiscal 2026 over fiscal 2025?

Gold answer: 65.47%

- [ ] `NVDA_FY2026_10K.pdf` page 139: "Revenue" = 215,938 (fiscal year ended January 25, 2026, in millions)
- [ ] `NVDA_FY2026_10K.pdf` page 139: "Revenue" = 130,497 (fiscal year ended January 26, 2025, in millions)

Derivation: 215,938 / 130,497 - 1 = 0.6547.

Note: Nvidia fiscal 2026 ended January 25, 2026, so most of it is calendar 2025. MD&A may print a rounded percent change.

## C24 (multi_hop)

Question: What share of its 2025 revenue did Meta spend on research and development?

Gold answer: 28.55%

- [ ] `META_FY2025_10K.pdf` page 90: "Research and development" = 57,372 (year ended December 31, 2025, in millions)
- [ ] `META_FY2025_10K.pdf` page 90: "Revenue" = 200,966 (year ended December 31, 2025, in millions)

Derivation: 57,372 / 200,966 = 0.2855.

## C25 (multi_hop)

Question: What was Pfizer's R&D spend as a percentage of total revenues in 2025?

Gold answer: 16.68%

- [ ] `PFE_FY2025_10K.pdf` page 58: "Research and development expenses" = 10,437 (year ended December 31, 2025, in millions)
- [ ] `PFE_FY2025_10K.pdf` page 58: "Total revenues" = 62,579 (year ended December 31, 2025, in millions)

Derivation: 10,437 / 62,579 = 0.1668.

Note: Excludes the separate Acquired in-process R&D line (1,613). Including it gives 0.1926; the question means the R&D expense line.

## C27 (multi_hop)

Question: Dividing fiscal 2026 membership fee revenue by year-end paid members, what did Costco earn per paid member?

Gold answer: About $70.24 per paid member

- [ ] `COST_FY2026_10K.pdf` page 38: "Membership fees" = 5,907 (52 weeks ended August 30, 2026, in millions)
- [ ] `COST_FY2026_10K.pdf` page 6: "Paid members (in thousands)" = 84,100 (end of fiscal 2026, in thousands)

Derivation: 5,907 million / 84,100 thousand = 5,907,000,000 / 84,100,000 = $70.24.

Note: Fees are in millions, members in thousands. Mixing scales gives 0.0702 or 70,238. Costco fiscal 2026 ended August 30, 2026.

## C29 (multi_hop)

Question: What was Mastercard's net profit margin in 2025?

Gold answer: 45.65%

- [ ] `MA_FY2025_10K.pdf` page 69: "Net Income" = 14,968 (year ended December 31, 2025, in millions)
- [ ] `MA_FY2025_10K.pdf` page 69: "Net Revenue" = 32,791 (year ended December 31, 2025, in millions)

Derivation: 14,968 / 32,791 = 0.4565.

Note: Mastercard sits in the banks_finance cluster.

## C31 (multi_hop)

Question: What share of Amazon's 2025 operating income came from AWS?

Gold answer: 57.03%

- [ ] `AMZN_FY2025_10K.pdf` page 68: "AWS Operating income" = 45,606 (year ended December 31, 2025, in millions)
- [ ] `AMZN_FY2025_10K.pdf` page 68: "Consolidated Operating income" = 79,975 (year ended December 31, 2025, in millions)

Derivation: 45,606 / 79,975 = 0.5703.

Note: Amazon prints columns oldest-first (2023, 2024, 2025). Consolidated operating income 79,975 is also on the income statement, PDF page 37.

## C32 (multi_hop)

Question: What was Netflix's operating margin in 2025?

Gold answer: Not answerable: Netflix is not one of the companies in this filing set.


Derivation: No Netflix filing is in the manifest; Netflix appears only in passing in other companies' filings, with no financials.

## C34 (cross_filing)

Question: Rank JPMorgan, Bank of America, Citigroup and Wells Fargo by deposits at the end of 2025, largest first.

Gold answer: JPMorgan ($2,559,320M) > Bank of America ($2,018,729M) > Wells Fargo ($1,426,207M) > Citigroup ($1,403,573M)

- [ ] `JPM_FY2025_10K.pdf` page 169: "Deposits" = 2,559,320 (December 31, 2025, in millions)
- [ ] `BAC_FY2025_10K.pdf` page 94: "Total deposits" = 2,018,729 (December 31, 2025, in millions)
- [ ] `WFC_FY2025_10K.pdf` page 93: "Total deposits" = 1,426,207 (December 31, 2025, in millions)
- [ ] `C_FY2025_10K.pdf` page 149: "Deposits" = 1,403,573 (December 31, 2025, in millions)

Derivation: Sort the four year-end deposit balances: 2,559,320 > 2,018,729 > 1,426,207 > 1,403,573.

Note: Wells Fargo edges Citi by about $22.6 billion; easy to get the last two backwards.

## C35 (cross_filing)

Question: Rank the six big banks in the set (JPMorgan, Bank of America, Citi, Wells Fargo, Goldman Sachs, Morgan Stanley) by 2025 net income attributable to the company.

Gold answer: JPMorgan (57,048) > Bank of America (30,509) > Wells Fargo (21,338) > Goldman Sachs (17,176) > Morgan Stanley (16,861) > Citigroup (14,306), $ millions

- [ ] `JPM_FY2025_10K.pdf` page 167: "Net income" = 57,048 (2025, in millions)
- [ ] `BAC_FY2025_10K.pdf` page 93: "Net income" = 30,509 (2025, in millions)
- [ ] `WFC_FY2025_10K.pdf` page 91: "Wells Fargo net income" = 21,338 (2025, in millions)
- [ ] `GS_FY2025_10K.pdf` page 141: "Net earnings" = 17,176 (2025, in millions)
- [ ] `MS_FY2025_10K.pdf` page 80: "Net income applicable to Morgan Stanley" = 16,861 (2025, in millions)
- [ ] `C_FY2025_10K.pdf` page 146: "Citigroup's net income" = 14,306 (2025, in millions)

Derivation: Sort the six attributable net income figures.

Note: Order is the same before noncontrolling interests (MS 17,025, Citi 14,452, WFC 21,358). Goldman vs Morgan Stanley is the close pair (315 apart).

## C36 (cross_filing)

Question: Which carmaker had the larger workforce at the end of 2025, Ford or GM, and by roughly how many people?

Gold answer: Ford, about 169,000 vs about 155,000 (14,000 more)

- [ ] `F_FY2025_10K.pdf` page 21: "Total Company employees (in thousands)" = 169 (December 31, 2025, in thousands)
- [ ] `GM_FY2025_10K.pdf` page 10: "Total Worldwide employment (in thousands)" = 155 (December 31, 2025, in thousands)

Derivation: Both tables are in thousands: Ford 169 thousand vs GM 155 thousand; 169,000 - 155,000 = 14,000.

Note: Ford's table prints 2024 first then 2025 (171 then 169): the 2025 figure is the right-hand column. GM's text says about 88,000 hourly plus 68,000 salaried (156,000); the table total is 155 thousand. Ford is larger either way.

## C37 (cross_filing)

Question: Among Merck, J&J and Pfizer, who spent the most on R&D in 2025? Rank all three.

Gold answer: Merck ($15,789M) > J&J ($14,665M) > Pfizer ($10,437M)

- [ ] `MRK_FY2025_10K.pdf` page 79: "Research and development" = 15,789 (2025, in millions)
- [ ] `JNJ_FY2025_10K.pdf` page 50: "Research and development expense" = 14,665 (fiscal 2025 (ended December 28, 2025), in millions)
- [ ] `PFE_FY2025_10K.pdf` page 58: "Research and development expenses" = 10,437 (2025, in millions)

Derivation: Sort the three R&D expense lines.

Note: Order holds if J&J's separate IPR&D impairments (81) and Pfizer's Acquired IPR&D (1,613) are added: J&J 14,746, Pfizer 12,050.

## C40 (cross_filing)

Question: Comparing each network's latest annual report in the set, did Visa or Mastercard earn more, and by how much?

Gold answer: Visa, by $8,536 million (23,504 for fiscal 2026 vs 14,968 for calendar 2025)

- [ ] `V_FY2026_10K.pdf` page 60: "Net income" = 23,504 (fiscal year ended September 30, 2026, in millions)
- [ ] `MA_FY2025_10K.pdf` page 69: "Net Income" = 14,968 (year ended December 31, 2025, in millions)

Derivation: 23,504 - 14,968 = 8,536 million.

Note: Periods differ: Visa fiscal 2026 ended September 30, 2026; Mastercard fiscal year = calendar 2025. Gap value 8,536,000,000 USD.

## C42 (cross_filing)

Question: How many of the 39 readable companies in the set ended their latest fiscal year with more than $1 trillion of total assets, and which are they?

Gold answer: Six: JPMorgan, Bank of America, Citigroup, Wells Fargo, Goldman Sachs and Morgan Stanley

- [ ] `AAPL_FY2026_10K.pdf` page 37: "Total assets" = 404,548 (September 26, 2026, in millions)
- [ ] `MSFT_FY2026_10K.pdf` page 52: "Total assets" = 758,376 (June 30, 2026, in millions)
- [ ] `GOOGL_FY2025_10K.pdf` page 49: "Total assets" = 595,281 (December 31, 2025, in millions)
- [ ] `AMZN_FY2025_10K.pdf` page 39: "Total assets" = 818,042 (December 31, 2025, in millions)
- [ ] `META_FY2025_10K.pdf` page 89: "Total assets" = 366,021 (December 31, 2025, in millions)
- [ ] `NVDA_FY2026_10K.pdf` page 141: "Total assets" = 206,803 (January 25, 2026, in millions)
- [ ] `AMD_FY2025_10K.pdf` page 63: "Total assets" = 76,926 (December 27, 2025, in millions)
- [ ] `ORCL_FY2026_10K.pdf` page 66: "Total assets" = 261,759 (May 31, 2026, in millions)
- [ ] `CRM_FY2026_10K.pdf` page 66: "Total assets" = 112,305 (January 31, 2026, in millions)
- [ ] `ADBE_FY2025_10K.pdf` page 47: "Total assets" = 29,496 (November 28, 2025, in millions)
- [ ] `CSCO_FY2026_10K.pdf` page 75: "Total assets" = 129,637 (July 25, 2026, in millions)
- [ ] `JPM_FY2025_10K.pdf` page 169: "Total assets" = 4,424,900 (December 31, 2025, in millions)
- [ ] `BAC_FY2025_10K.pdf` page 94: "Total assets" = 3,411,738 (December 31, 2025, in millions)
- [ ] `C_FY2025_10K.pdf` page 148: "Total assets" = 2,657,202 (December 31, 2025, in millions)
- [ ] `WFC_FY2025_10K.pdf` page 93: "Total assets" = 2,148,631 (December 31, 2025, in millions)
- [ ] `GS_FY2025_10K.pdf` page 142: "Total assets" = 1,809,320 (December 31, 2025, in millions)
- [ ] `MS_FY2025_10K.pdf` page 81: "Total assets" = 1,420,270 (December 31, 2025, in millions)
- [ ] `V_FY2026_10K.pdf` page 59: "Total assets" = 94,360 (September 30, 2026, in millions)
- [ ] `MA_FY2025_10K.pdf` page 71: "Total assets" = 54,157 (December 31, 2025, in millions)
- [ ] `AXP_FY2025_10K.pdf` page 97: "Total assets" = 300,052 (December 31, 2025, in millions)
- [ ] `BLK_FY2025_10K.pdf` page 79: "Total assets" = 169,998 (December 31, 2025, in millions)
- [ ] `WMT_FY2026_10K.pdf` page 54: "Total assets" = 284,668 (January 31, 2026, in millions)
- [ ] `COST_FY2026_10K.pdf` page 40: "Total assets" = 89,045 (August 30, 2026, in millions)
- [ ] `TGT_FY2025_10K.pdf` page 53: "Total assets" = 59,490 (January 31, 2026, in millions)
- [ ] `KO_FY2025_10K.pdf` page 65: "Total assets" = 104,816 (December 31, 2025, in millions)
- [ ] `PEP_FY2025_10K.pdf` page 64: "Total assets" = 107,399 (December 27, 2025, in millions)
- [ ] `PG_FY2026_10K.pdf` page 40: "Total assets" = 126,521 (June 30, 2026, in millions)
- [ ] `NKE_FY2026_10K.pdf` page 60: "Total assets" = 38,410 (May 31, 2026, in millions)
- [ ] `HD_FY2025_10K.pdf` page 47: "Total assets" = 105,095 (February 1, 2026, in millions)
- [ ] `JNJ_FY2025_10K.pdf` page 49: "Total assets" = 199,210 (December 28, 2025, in millions)
- [ ] `PFE_FY2025_10K.pdf` page 60: "Total assets" = 208,160 (December 31, 2025, in millions)
- [ ] `MRK_FY2025_10K.pdf` page 80: "Total assets (unlabeled total line under Other Assets)" = 136,866 (December 31, 2025, in millions)
- [ ] `UNH_FY2025_10K.pdf` page 43: "Total assets" = 309,581 (December 31, 2025, in millions)
- [ ] `TSLA_FY2025_10K.pdf` page 52: "Total assets" = 137,806 (December 31, 2025, in millions)
- [ ] `F_FY2025_10K.pdf` page 116: "Total assets" = 289,160 (December 31, 2025, in millions)
- [ ] `GM_FY2025_10K.pdf` page 58: "Total assets" = 281,284 (December 31, 2025, in millions)
- [ ] `XOM_FY2025_10K.pdf` page 75: "Total assets" = 448,980 (December 31, 2025, in millions)
- [ ] `CVX_FY2025_10K.pdf` page 68: "Total assets" = 324,012 (December 31, 2025, in millions)
- [ ] `CAT_FY2025_10K.pdf` page 63: "Total assets" = 98,585 (December 31, 2025, in millions)

Derivation: Count total assets above 1,000,000 million among the 39: JPM 4,424,900; BAC 3,411,738; Citi 2,657,202; WFC 2,148,631; GS 1,809,320; MS 1,420,270. Largest non-bank is Amazon at 818,042.

Note: Members: JPMorgan Chase, Bank of America, Citigroup, Wells Fargo, Goldman Sachs, Morgan Stanley. McDonald's excluded (unreadable).

## C44 (cross_filing)

Question: What 2024 net income did Bank of America originally report, what does its 2025 annual report now show for 2024, and why the difference?

Gold answer: Originally $27,132 million; the 2025 10-K shows $26,973 million, $159 million lower, because BofA changed its accounting for tax-related equity investments (proportional amortization) and applied it retrospectively.

- [ ] `BAC_FY2024_10K.pdf` page 92: "Net income" = 27,132 (year ended December 31, 2024 (as filed in FY2024 10-K), in millions)
- [ ] `BAC_FY2025_10K.pdf` page 93: "Net income" = 26,973 (year ended December 31, 2024 (restated comparative in FY2025 10-K), in millions)
- [ ] `BAC_FY2025_10K.pdf` page 97: "Accounting change applied retrospectively (tax-related equity investments)" = applied  retrospectively (effective fourth quarter 2025, in text)

Derivation: 26,973 - 27,132 = -159 million.

Note: Other 2024 lines moved more: total revenue net of interest expense 101,887 -> 105,856 (+3,969); income tax expense 2,122 -> 6,250 (+4,128). answer_value is the change in net income.

## C45 (cross_filing)

Question: Apple's September 2025 balance sheet appears in both of its last two annual reports. Did the 'other non-current assets' figure for that date change, and if so by how much and why?

Gold answer: Yes. It fell from $83,727 million to $72,634 million (down $11,093 million) because intangible assets, net ($11,093 million) are now shown on their own line; total assets are unchanged at $359,241 million.

- [ ] `AAPL_FY2025_10K.pdf` page 34: "Other non-current assets" = 83,727 (September 27, 2025 (as filed in FY2025 10-K), in millions)
- [ ] `AAPL_FY2026_10K.pdf` page 37: "Other non-current assets" = 72,634 (September 27, 2025 (comparative in FY2026 10-K), in millions)
- [ ] `AAPL_FY2026_10K.pdf` page 37: "Intangible assets, net" = 11,093 (September 27, 2025 (comparative in FY2026 10-K), in millions)
- [ ] `AAPL_FY2026_10K.pdf` page 40: "Reclassification note" = reclassified (Note 1, in text)

Derivation: 72,634 - 83,727 = -11,093 = minus the new Intangible assets, net line (11,093). 72,634 + 11,093 = 83,727.

Note: A reclassification, not a change in totals. Note 1 says certain prior period amounts were reclassified to conform to current presentation.

## C46 (cross_filing)

Question: Did Coca-Cola's latest 10-K change the year-end 2024 balance for prepaid expenses and other current assets versus what it first reported?

Gold answer: Yes. It went from $3,129 million to $2,998 million (down $131 million); the $131 million now appears as a separate Assets held for sale line. Total current assets stay $25,997 million.

- [ ] `KO_FY2024_10K.pdf` page 66: "Prepaid expenses and other current assets" = 3,129 (December 31, 2024 (as filed in FY2024 10-K), in millions)
- [ ] `KO_FY2025_10K.pdf` page 65: "Prepaid expenses and other current assets" = 2,998 (December 31, 2024 (comparative in FY2025 10-K), in millions)
- [ ] `KO_FY2025_10K.pdf` page 65: "Assets held for sale" = 131 (December 31, 2024 (comparative in FY2025 10-K), in millions)

Derivation: 2,998 - 3,129 = -131 = minus Assets held for sale (131) in the 2024 comparative column.

Note: Same pattern on the liability side: Accounts payable and accrued expenses 21,715 -> 21,712 with 3 shown as Liabilities held for sale. KO notes prior-year amounts were revised to conform to current presentation (PDF page 68).

## C48 (cross_filing)

Question: How did Intel's 2025 revenue compare with AMD's?

Gold answer: Not answerable: Intel is not in this filing set. AMD's 2025 net revenue was $34,639 million, but no Intel figures are available.

- [ ] `AMD_FY2025_10K.pdf` page 61: "Net revenue" = 34,639 (year ended December 27, 2025, in millions)

Derivation: Only one side of the comparison exists in the set. AMD's 10-K names Intel as a competitor but gives no Intel financials.

Note: A partial answer giving AMD's figure is fine; any Intel number is a hallucination.
