"""Write the training set for the local planner: (question, plan) pairs in plan language v2.

A plan is sampled first from the graph's own vocabulary (companies, concepts,
metrics, attributes, and line labels the filings print), then put into words
with a frame, a way of naming the company, the measure and the period. Every
plan is checked with `tieout.plan.validate` before it is kept.

Validation rows use frames and phrasings that never appear in training, so the
validation score says something about wording the model has not seen. The real
test is the benchmark, whose questions were written by people reading the PDFs.

Every operation of the language is covered: get (concept, metric, item, quarter,
year to date, first reported), calc (difference, ratio, sum, per employee, per
unit, share of, size of a loss), change, rank (by concept, metric, item,
attribute, growth, ratio; one, several or two named companies), filter (by
measure, by growth or change, by text attribute) with count, agg, series,
members with rank, filter, count and a read of the chosen line, attr, restated,
and "cannot" for companies outside the set and facts filings do not print.

Usage: python finetune/make_data.py [--train 9000] [--val 600] [--seed 7]
Writes finetune/data_v2/train.jsonl, val.jsonl and finetune/system_prompt.v2.txt
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import pathlib
import random
import re
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tieout.concepts import concept_defs  # noqa: E402
from tieout.graph import Graph  # noqa: E402
from tieout.plan import CLUSTERS, metric_defs, validate  # noqa: E402

ALIASES = yaml.safe_load((ROOT / "semantics" / "company_aliases.yaml").read_text(encoding="utf-8"))
OUT = ROOT / "finetune" / "data_v2"
STOCKS = {n for n, d in concept_defs().items() if d.nature == "stock"}
BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS"]
CARDS = ["V", "MA", "AXP"]
RETAIL = ["WMT", "COST", "TGT", "HD", "NKE", "MCD", "KO", "PEP", "PG"]
TECH = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "ORCL", "CRM", "ADBE", "CSCO"]
INDUSTRIAL = ["CAT", "F", "GM", "TSLA", "XOM", "CVX"]
HEALTH = ["JNJ", "PFE", "MRK", "UNH"]
NOT_LISTED = ["Netflix", "Intel", "Boeing", "Starbucks", "U.S. Bancorp", "Comcast", "Disney", "Verizon", "AT&T",
              "PayPal", "Qualcomm", "IBM", "Broadcom", "UPS", "FedEx", "AbbVie", "Eli Lilly", "Charles Schwab",
              "Capital One", "Kroger", "Lowe's", "Uber", "Texas Instruments", "Honeywell", "Deere", "ConocoPhillips"]

# (training phrasings, phrasings used only in validation)
CONCEPT_WORDS = {
    "Revenue": (["revenue", "total revenue", "net sales", "sales", "total net sales", "revenues", "top line"], ["turnover", "total sales"]),
    "CostOfRevenue": (["cost of revenue", "cost of sales", "cost of goods sold"], ["COGS"]),
    "GrossProfit": (["gross profit"], ["gross income"]),
    "ResearchAndDevelopment": (["research and development expense", "R&D spending", "R&D expense", "R&D"], ["research spend"]),
    "SellingGeneralAdmin": (["selling, general and administrative expense", "SG&A", "SG&A expense"], ["overhead (SG&A)"]),
    "SalesAndMarketing": (["sales and marketing expense", "sales and marketing spend"], ["marketing and sales costs"]),
    "GeneralAndAdministrative": (["general and administrative expense", "G&A expense"], ["G&A costs"]),
    "OperatingIncome": (["operating income", "income from operations", "operating profit"], ["operating earnings"]),
    "PretaxIncome": (["pre-tax income", "income before income taxes", "income before taxes"], ["earnings before tax"]),
    "IncomeTax": (["income tax expense", "provision for income taxes", "tax provision"], ["income taxes"]),
    "InterestExpense": (["interest expense"], ["interest cost"]),
    "NetIncome": (["net income", "net earnings", "profit", "net profit", "net income attributable to shareholders"],
                  ["bottom line", "earnings attributable to the company"]),
    "ConsolidatedNetIncome": (["consolidated net income including noncontrolling interests", "net income including minority interests"],
                              ["net income before noncontrolling interests"]),
    "EPSDiluted": (["diluted earnings per share", "diluted EPS"], ["EPS on a diluted basis"]),
    "EPSBasic": (["basic earnings per share", "basic EPS"], ["EPS on a basic basis"]),
    "SharesDiluted": (["diluted share count", "diluted weighted-average shares"], ["shares used for diluted EPS"]),
    "SharesBasic": (["basic weighted-average shares", "basic share count"], ["shares used for basic EPS"]),
    "NetInterestIncome": (["net interest income"], ["NII"]),
    "NoninterestIncome": (["noninterest income", "total noninterest income"], ["fee income outside interest"]),
    "ProvisionForCreditLosses": (["provision for credit losses", "credit loss provision"], ["loan loss provision"]),
    "NoninterestExpense": (["noninterest expense", "total noninterest expense"], ["non-interest costs"]),
    "Cash": (["cash and cash equivalents", "cash"], ["cash on hand"]),
    "TotalCurrentAssets": (["total current assets", "current assets"], ["short-term assets"]),
    "TotalAssets": (["total assets", "assets"], ["balance sheet size"]),
    "TotalCurrentLiabilities": (["total current liabilities", "current liabilities"], ["short-term liabilities"]),
    "TotalLiabilities": (["total liabilities", "liabilities"], ["total obligations"]),
    "ShareholdersEquity": (["shareholders' equity", "stockholders' equity", "book value of equity"], ["equity attributable to shareholders"]),
    "TotalEquity": (["total equity including noncontrolling interests"], ["equity including minority interests"]),
    "TotalLiabilitiesAndEquity": (["total liabilities and equity"], ["liabilities plus equity"]),
    "AccountsReceivable": (["accounts receivable", "receivables"], ["trade receivables"]),
    "PropertyPlantEquipment": (["property, plant and equipment", "net PP&E"], ["fixed assets"]),
    "IntangibleAssets": (["intangible assets", "intangibles other than goodwill"], ["acquired intangibles"]),
    "AccountsPayable": (["accounts payable", "payables"], ["trade payables"]),
    "RetainedEarnings": (["retained earnings"], ["accumulated earnings"]),
    "Loans": (["total loans", "loans"], ["loan book"]),
    "Inventory": (["inventory", "inventories"], ["stock of inventory"]),
    "Goodwill": (["goodwill"], ["goodwill balance"]),
    "LongTermDebt": (["long-term debt"], ["non-current debt"]),
    "Deposits": (["total deposits", "deposits"], ["customer deposits"]),
    "OperatingCashFlow": (["operating cash flow", "cash from operations", "net cash provided by operating activities"], ["cash generated by the business"]),
    "InvestingCashFlow": (["investing cash flow", "net cash used in investing activities"], ["cash used for investing"]),
    "FinancingCashFlow": (["financing cash flow", "net cash used in financing activities"], ["cash from financing"]),
    "CapitalExpenditures": (["capital expenditures", "capex", "purchases of property and equipment"], ["capital spending"]),
    "DepreciationAmortization": (["depreciation and amortization", "D&A"], ["depreciation charge"]),
    "StockBasedCompensation": (["stock-based compensation", "share-based compensation"], ["equity compensation expense"]),
    "DividendsPaid": (["dividends paid", "cash dividends"], ["what it paid out in dividends"]),
    "ShareRepurchases": (["share repurchases", "stock buybacks", "buybacks"], ["what it spent buying back stock"]),
}
METRIC_WORDS = {
    "gross_margin": (["gross margin", "gross margin percentage"], ["gross profit margin"]),
    "gross_profit": (["gross profit"], ["gross income"]),
    "operating_margin": (["operating margin"], ["operating profit margin"]),
    "net_margin": (["net margin", "net profit margin"], ["profit margin"]),
    "rnd_intensity": (["R&D as a share of revenue", "R&D intensity"], ["R&D-to-sales ratio"]),
    "effective_tax_rate": (["effective tax rate"], ["tax rate"]),
    "free_cash_flow": (["free cash flow"], ["FCF"]),
    "capex_intensity": (["capex as a share of revenue", "capital intensity"], ["capex-to-sales ratio"]),
    "return_on_equity": (["return on equity", "ROE"], ["return on shareholders' equity"]),
    "return_on_assets": (["return on assets", "ROA"], ["return on total assets"]),
    "current_ratio": (["current ratio"], ["ratio of current assets to current liabilities"]),
    "total_liabilities": (["total liabilities"], ["total obligations"]),
    "liabilities_to_equity": (["liabilities-to-equity ratio", "total liabilities to equity"], ["leverage (liabilities over equity)"]),
    "asset_turnover": (["asset turnover"], ["sales per dollar of assets"]),
    "cash_conversion": (["cash conversion (operating cash flow over net income)"], ["operating cash flow relative to net income"]),
    "shareholder_payout": (["total shareholder payout (dividends plus buybacks)"], ["cash returned to shareholders"]),
    "sbc_share_of_revenue": (["stock-based compensation as a share of revenue"], ["SBC-to-revenue ratio"]),
}
CLUSTER_WORDS = {
    "big_tech": (["big tech", "technology"], ["tech"]),
    "banks_finance": (["bank or financial", "financial"], ["banking and finance"]),
    "consumer_retail": (["consumer or retail", "consumer"], ["retail and consumer goods"]),
    "health": (["health care", "healthcare"], ["pharma and health"]),
    "industrial_auto_energy": (["industrial, auto or energy"], ["industrial and energy"]),
}
ATTR_Q = {
    "auditor": (["Who audits {co}?", "Who is {co}'s auditor?", "Which accounting firm audits {co}?"], ["Name the independent auditor of {co}."]),
    "auditor_since": (["Since what year has {co} had its current auditor?"], ["How long has {co}'s auditor served (since which year)?"]),
    "state_of_incorporation": (["In which state is {co} incorporated?", "Where is {co} incorporated?"], ["What is {co}'s state of incorporation?"]),
    "hq_state": (["In which state is {co} headquartered?"], ["Which state hosts {co}'s head office?"]),
    "hq_city": (["In which city is {co} headquartered?"], ["Where is {co}'s head office (city)?"]),
    "exchange": (["On which exchange is {co} listed?"], ["Which stock exchange lists {co}'s shares?"]),
    "employees": (["How many employees does {co} have?", "What is {co}'s headcount?", "Roughly how many people does {co} employ?"],
                  ["How large is {co}'s workforce?"]),
    "ceo": (["Who is the CEO of {co}?"], ["Who runs {co} as chief executive?"]),
    "cfo": (["Who is the CFO of {co}?"], ["Who is {co}'s chief financial officer?"]),
    "shares_outstanding": (["How many shares does {co} have outstanding?"], ["What is {co}'s share count outstanding?"]),
    "fiscal_year_end": (["When does {co}'s fiscal year end?"], ["On what date does {co} close its fiscal year?"]),
}
HOP_ATTR = {"auditor": ["Who audits", "Which firm audits"], "state_of_incorporation": ["In which state is incorporated"],
            "hq_state": ["Which state is home to"], "ceo": ["Who is the CEO of"], "employees": ["How many people work at"]}
AUDITORS = ["Deloitte", "PricewaterhouseCoopers", "Ernst & Young", "KPMG"]
STATES = ["Delaware", "New York", "California", "Washington", "Minnesota", "Oregon", "Texas"]
CANNOT = [
    ("What was {co}'s market share in China in {yr}?", "market share is not a figure the filings print"),
    ("What is {co}'s current stock price?", "the graph holds filing facts, not market prices"),
    ("What is {co}'s market capitalization today?", "the graph holds filing facts, not market prices"),
    ("What does {co}'s management expect revenue to be next year?", "the graph holds reported facts, not forecasts"),
    ("What credit rating does {co} have?", "credit ratings are not reported figures"),
    ("How much did {co} pay its CEO in {yr}?", "executive pay is reported in the proxy statement, not the annual report"),
    ("How much did {co} pay its auditor in audit fees for {yr}?", "audit fees are reported in the proxy statement, not the annual report"),
    ("What will {co}'s earnings per share be next year?", "the graph holds reported facts, not forecasts"),
]
STATES_PLACES = ["California", "Texas", "Florida", "New York", "Illinois", "Ohio", "Georgia", "Canada", "Mexico", "Japan",
                 "the United Kingdom", "Germany", "China", "Puerto Rico", "Washington"]
# item families outside the three statements, by the kind of company that prints them
ITEM_FAMILIES = {
    "retail_count": (["WMT", "COST", "TGT", "HD", "MCD", "NKE"],
                     ["number of stores", "number of stores in {place}", "number of warehouses", "number of warehouses in {place}",
                      "number of restaurants", "number of restaurants in {place}", "total retail square feet", "paid members",
                      "number of Supercenters in the U.S.", "number of employees in the United States"]),
    "retail_usd": (["WMT", "COST", "TGT", "HD", "NKE", "MCD", "KO", "PEP", "PG"],
                   ["membership fee revenue", "advertising expense", "e-commerce sales", "franchised revenues",
                    "comparable sales growth", "International segment net sales", "online sales"]),
    "bank": (["JPM", "BAC", "C", "WFC", "GS", "MS"],
             ["CET1 capital ratio", "Tier 1 leverage ratio", "CET1 ratio under the standardized approach", "client assets",
              "net new assets", "investment banking fees", "trading revenue", "assets under management",
              "Wealth Management net revenues", "compensation and benefits", "average loans", "net charge-offs",
              "allowance for credit losses", "book value per share"]),
    "asset_cards": (["BLK", "V", "MA", "AXP"],
                    ["assets under management", "net inflows", "processed transactions", "payments volume",
                     "client incentives", "card member loans", "gross revenues", "billed business", "base fees"]),
    "tech": (["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "ORCL", "CRM", "ADBE", "CSCO"],
             ["remaining performance obligation", "annualized recurring revenue", "subscription revenue", "cloud revenue",
              "advertising revenue", "daily active people", "deferred revenue", "backlog", "services revenue",
              "Data Center revenue", "Gaming revenue", "product revenue"]),
    "industrial": (["CAT", "F", "GM", "TSLA", "XOM", "CVX"],
                   ["order backlog", "vehicles delivered", "consumer vehicle deliveries", "wholesale vehicle units",
                    "net oil-equivalent production", "Upstream capital and exploration expenditures", "segment EBIT",
                    "automotive sales revenue", "Financial Products revenues", "energy generation and storage revenue"]),
    "health": (["JNJ", "PFE", "MRK", "UNH"],
               ["Innovative Medicine sales", "MedTech sales", "Keytruda sales", "medical costs", "medical care ratio",
                "people served", "Optum revenues", "premiums revenue", "Pharmaceutical segment sales"]),
}
BREAKDOWNS = ["revenue by segment", "net sales by segment", "operating income by segment", "revenue by product",
              "net sales by product category", "revenue by region", "net sales by geographic segment",
              "revenues by business", "segment operating profit", "sales by product line", "revenue by service offering",
              "earnings from operations by segment", "pre-tax earnings by business segment", "net revenues by segment"]
BREAKDOWN_NOUN = {"segment": "segments", "product": "products", "region": "regions", "business": "businesses",
                  "line": "product lines", "offering": "service offerings", "category": "product categories"}


def _raw_rows() -> list:
    """Line labels as the filings print them, from the committed raw layer (read-only)."""
    try:
        blob = subprocess.run(["git", "show", "HEAD:graph/raw.json.gz"], cwd=ROOT, capture_output=True, check=True).stdout
        return json.load(gzip.open(io.BytesIO(blob), "rt", encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a fresh clone without history still works, with fewer item names
        return []


def _fix_possessives(text: str) -> str:
    """"McDonald's's" -> "McDonald's"; "a asset" -> "an asset"."""
    text = re.sub(r"(\w's)'s\b", r"\1", text)
    return re.sub(r"\ba ([aeiouAEIOU])", r"an \1", text)


def _clean(label: str) -> bool:
    return (3 < len(label) <= 36 and len(label.split()) <= 5 and not re.search(r"\d|\(|\)|:|;|%|\$", label)
            and not re.search(r"(?i)eliminat|intersegment|adjust|reconcil|unallocated|total|other|corporate|net of", label))


def _parts() -> dict:
    """Per company: labels of lines that add up to its revenue or operating income (segments, products, regions)."""
    out: dict = {}
    for r in _raw_rows():
        if r.get("anchor") in ("Revenue", "OperatingIncome") and r.get("role") == "operand" and r.get("unit") == "USD" \
                and _clean(r["label"].strip()):
            out.setdefault(r["ticker"], set()).add((r["label"].strip(), r["anchor"]))
    return {t: sorted(v) for t, v in out.items() if v}


PARTS = _parts()


class Gen:
    def __init__(self, seed: int, val: bool):
        self.r = random.Random(seed)
        self.val = val
        self.tickers = list(ALIASES)

    # ---------------------------------------------------------- wording
    def pick(self, pair):
        train, held = pair
        return self.r.choice(held if (self.val and held and self.r.random() < 0.6) else train)

    def co(self, t):
        return self.r.choice(ALIASES[t])

    def concept(self, pool=None):
        c = self.r.choice(pool or list(CONCEPT_WORDS))
        return c, self.pick(CONCEPT_WORDS[c])

    def metric(self, ratio_only=False):
        names = [m for m in METRIC_WORDS if not ratio_only or metric_defs()[m]["unit"] == "ratio"]
        m = self.r.choice(names)
        return m, self.pick(METRIC_WORDS[m])

    def plain_concepts(self):
        return [c for c in CONCEPT_WORDS if c not in ("NetInterestIncome", "NoninterestIncome", "ProvisionForCreditLosses",
                                                      "NoninterestExpense", "Deposits", "Loans", "SharesDiluted", "SharesBasic",
                                                      "TotalLiabilitiesAndEquity", "ConsolidatedNetIncome", "TotalEquity")]

    def measure(self, allow_item=False, allow_attr=False):
        """A measure and its words: concept, metric, item, growth, ratio or attribute."""
        x = self.r.random()
        if allow_attr and x < 0.08:
            return {"attr": "employees"}, self.r.choice(["number of employees", "headcount", "workforce"])
        if x < 0.40:
            c, w = self.concept(self.plain_concepts())
            return {"concept": c}, w
        if x < 0.68:
            m, w = self.metric()
            return {"metric": m}, w
        if x < 0.82:
            c, w = self.concept(["Revenue", "NetIncome", "OperatingIncome", "OperatingCashFlow", "EPSDiluted", "CapitalExpenditures"])
            return {"growth": {"concept": c}}, self.r.choice([f"{w} growth", f"growth in {w}", f"year-over-year {w} growth"])
        if x < 0.92 or not allow_item:
            (c1, w1), (c2, w2) = self.concept(["Inventory", "AccountsReceivable", "LongTermDebt", "Cash", "CapitalExpenditures",
                                               "ResearchAndDevelopment", "SellingGeneralAdmin", "DividendsPaid"]), \
                self.concept(["Revenue", "CostOfRevenue", "TotalAssets", "NetIncome", "OperatingCashFlow"])
            return {"ratio": [{"concept": c1}, {"concept": c2}]}, f"{w1} relative to {w2}"
        fam = self.r.choice(["bank", "retail_count", "tech", "industrial"])
        it = self.item_phrase(fam)
        return {"item": it}, it

    def item_phrase(self, family):
        _, phrases = ITEM_FAMILIES[family]
        return self.r.choice(phrases).replace("{place}", self.r.choice(STATES_PLACES))

    def item_for(self, t=None):
        """(ticker, item phrase, words) for a line outside the statements that this kind of company prints."""
        x = self.r.random()
        if x < 0.45 and PARTS:
            t = t if t in PARTS else self.r.choice(sorted(PARTS))
            label, anchor = self.r.choice(PARTS[t])
            noun = "revenue" if anchor == "Revenue" else "operating income"
            if anchor == "Revenue":
                noun = self.r.choice(["revenue", "net sales", "revenues", "sales"])
            item = f"{label} {noun}"
            return t, item, self.r.choice([f"{label} {noun}", f"{noun} from {label}", f"the {label} business's {noun}"])
        fams = [f for f, (cos, _) in ITEM_FAMILIES.items() if t is None or t in cos] or list(ITEM_FAMILIES)
        fam = self.r.choice(fams)
        cos, _ = ITEM_FAMILIES[fam]
        t = t if t in cos else self.r.choice(cos)
        it = self.item_phrase(fam)
        words = it if not it.startswith("number of ") else self.r.choice([it, "how many " + it[len("number of "):]])
        return t, it, words

    def period(self, stock=False):
        x = self.r.random()
        if x < 0.18:
            return {"fy": "latest"}, self.r.choice(["in its latest fiscal year", "in the most recent fiscal year", "most recently",
                                                    "for its last reported fiscal year"] if not stock else
                                                   ["at its latest fiscal year end", "at the end of its most recent fiscal year"])
        y = self.r.choice([2023, 2024, 2025, 2025, 2026, 2026])
        if stock:
            w = self.r.choice([f"at the end of fiscal {y}", f"at fiscal year-end {y}", f"as of the end of {y}", f"at the close of FY{y}",
                               f"going out of fiscal {y}"] + ([f"on its fiscal {y} balance sheet", f"at year-end {y}"] if self.val else []))
        else:
            w = self.r.choice([f"in fiscal {y}", f"in FY{y}", f"in {y}", f"for fiscal year {y}", f"during fiscal {y}", f"for {y}",
                               f"in its {y} fiscal year"] + ([f"over FY{str(y)[2:]}", f"across fiscal {y}", f"for the {y} fiscal year"]
                                                             if self.val else []))
        return {"fy": y}, w

    def quarter(self):
        y, q = self.r.choice([2025, 2026]), self.r.choice([1, 2, 3, 4])
        name = ["first", "second", "third", "fourth"][q - 1]
        return {"fy": y, "q": q}, self.r.choice([f"in the {name} quarter of fiscal {y}", f"in Q{q} of fiscal {y}", f"in Q{q} FY{y}",
                                                 f"in its fiscal {name} quarter of {y}", f"for the {name} quarter of {y}"])

    def ytd(self):
        y, q = self.r.choice([2025, 2026]), self.r.choice([2, 3])
        months = {2: "six", 3: "nine"}[q]
        return {"fy": y, "q": q, "ytd": True}, self.r.choice([f"in the first {months} months of fiscal {y}",
                                                              f"year to date through Q{q} of fiscal {y}",
                                                              f"for the {months} months of fiscal {y}"])

    def a_set(self):
        x = self.r.random()
        if x < 0.4:
            cl = self.r.choice(list(CLUSTERS))
            return {"cluster": cl}, f"{self.pick(CLUSTER_WORDS[cl])} company", f"{self.pick(CLUSTER_WORDS[cl])} companies"
        if x < 0.6:
            return "all", "company", "companies"
        pool = self.r.choice([BANKS, CARDS + ["BLK"], RETAIL, TECH, INDUSTRIAL, HEALTH, self.tickers])
        ts = self.r.sample(pool, min(len(pool), self.r.choice([3, 4, 5])))
        names = [self.co(t) for t in ts]
        return ts, None, ", ".join(names[:-1]) + " and " + names[-1]

    @staticmethod
    def one(step):
        return {"steps": [dict(id="s1", **step)], "answer": "s1"}

    # ------------------------------------------------------------ plans
    def get(self):
        c, w = self.concept()
        t = self.r.choice(BANKS) if c in ("NetInterestIncome", "NoninterestIncome", "ProvisionForCreditLosses",
                                          "NoninterestExpense", "Deposits", "Loans") else self.r.choice(self.tickers)
        x = self.r.random()
        if x < 0.14 and c not in STOCKS:
            p, pw = self.quarter()
        elif x < 0.20 and c not in STOCKS:
            p, pw = self.ytd()
        else:
            p, pw = self.period(stock=c in STOCKS)
        frames = ["What was {co}'s {w} {pw}?", "How much {w} did {co} report {pw}?", "What did {co} report as {w} {pw}?",
                  "Give me {co}'s {w} {pw}.", "{co} {w} {pw}?", "Tell me the {w} of {co} {pw}.", "What {w} did {co} post {pw}?"]
        if self.val:
            frames = ["I need {co}'s {w} {pw}.", "Look up the {w} that {co} booked {pw}.", "What figure did {co} show for {w} {pw}?"]
        q = self.r.choice(frames).format(co=self.co(t), w=w, pw=pw)
        return q, self.one({"op": "get", "company": t, "concept": c, "period": p})

    def first_reported(self):
        c, w = self.concept(self.plain_concepts())
        t, y = self.r.choice(self.tickers), self.r.choice([2024, 2025])
        q = self.r.choice([f"What did {self.co(t)} originally report as {w} for fiscal {y}, before any later revision?",
                           f"In its fiscal {y} annual report, what {w} did {self.co(t)} report for fiscal {y}?",
                           f"As first reported, what was {self.co(t)}'s {w} for {y}?"])
        return q, self.one({"op": "get", "company": t, "concept": c, "period": {"fy": y}, "as": "first_reported"})

    def metric_q(self):
        m, w = self.metric()
        t = self.r.choice(self.tickers)
        p, pw = self.period()
        frames = ["What was {co}'s {w} {pw}?", "Compute {co}'s {w} {pw}.", "What {w} did {co} achieve {pw}?", "{co}'s {w} {pw}?"]
        if self.val:
            frames = ["Work out the {w} for {co} {pw}.", "How did {co} do on {w} {pw}?"]
        q = self.r.choice(frames).format(co=self.co(t), w=w, pw=pw)
        return q, self.one({"op": "metric", "company": t, "metric": m, "period": p})

    def item_q(self):
        t, item, words = self.item_for()
        stock = any(k in item for k in ("number of", "assets", "backlog", "obligation", "members", "ratio", "loans", "allowance"))
        p, pw = self.period(stock=stock)
        if self.r.random() < 0.15 and not stock:
            p, pw = self.quarter()
        if words.startswith("how many "):
            q = self.r.choice([f"{words[0].upper() + words[1:]} did {self.co(t)} have {pw}?",
                               f"{words[0].upper() + words[1:]} did {self.co(t)} operate {pw}?"])
        else:
            frames = ["What was {co}'s {w} {pw}?", "How much {w} did {co} report {pw}?", "What {w} did {co} disclose {pw}?",
                      "{co}: {w} {pw}?", "How large was {co}'s {w} {pw}?"]
            if self.val:
                frames = ["How big was {co}'s {w} {pw}?", "Give me the {w} line for {co} {pw}."]
            q = self.r.choice(frames).format(co=self.co(t), w=words, pw=pw)
        return q, self.one({"op": "get", "company": t, "item": item, "period": p})

    def change(self):
        x = self.r.random()
        t = self.r.choice(self.tickers)
        if x < 0.25:
            t, item, words = self.item_for()
            by, w = {"item": item}, words
        elif x < 0.6:
            c, w = self.concept(self.plain_concepts())
            by = {"concept": c}
        else:
            m, w = self.metric()
            by = {"metric": m}
        mode = "pct" if self.r.random() < 0.7 else "abs"
        if self.r.random() < 0.3:
            p1, p2 = {"fy": "prior"}, {"fy": "latest"}
            pw = self.r.choice(["in its most recent fiscal year", "over the latest fiscal year", "year over year in its latest fiscal year"])
        else:
            y = self.r.choice([2024, 2025, 2026])
            p1, p2 = {"fy": y - 1}, {"fy": y}
            pw = self.r.choice([f"from fiscal {y - 1} to fiscal {y}", f"in fiscal {y} compared with the year before",
                                f"between FY{y - 1} and FY{y}", f"in fiscal {y}"])
        if mode == "pct":
            frames = ["By what percentage did {co}'s {w} change {pw}?", "How fast did {co}'s {w} grow {pw}?",
                      "What was the growth rate of {co}'s {w} {pw}?"]
            if self.val:
                frames = ["What was the year-over-year percentage move in {co}'s {w} {pw}?"]
        else:
            frames = ["By how much did {co}'s {w} change {pw}?", "What was the dollar change in {co}'s {w} {pw}?"]
            if self.val:
                frames = ["How many dollars did {co}'s {w} move by {pw}?"]
        q = self.r.choice(frames).format(co=self.co(t), w=w, pw=pw)
        return q, self.one({"op": "change", "company": t, **by, "from": p1, "to": p2, "mode": mode})

    def compare_calc(self):
        c, w = self.concept(self.plain_concepts())
        a, b = self.r.sample(self.tickers, 2)
        p, pw = self.period(stock=c in STOCKS)
        kind = self.r.choice(["diff", "ratio", "sum"])
        if kind == "diff":
            frames = ["How much higher was {a}'s {w} than {b}'s {pw}?", "What is the difference between {a}'s and {b}'s {w} {pw}?",
                      "How much more {w} did {a} have than {b} {pw}?"]
            expr = "s1 - s2"
        elif kind == "ratio":
            frames = ["How many times larger was {a}'s {w} than {b}'s {pw}?", "What is the ratio of {a}'s {w} to {b}'s {pw}?"]
            expr = "s1 / s2"
        else:
            frames = ["What was the combined {w} of {a} and {b} {pw}?", "Add up {a}'s and {b}'s {w} {pw}."]
            expr = "s1 + s2"
        q = self.r.choice(frames).format(a=self.co(a), b=self.co(b), w=w, pw=pw)
        return q, {"steps": [{"id": "s1", "op": "get", "company": a, "concept": c, "period": p},
                             {"id": "s2", "op": "get", "company": b, "concept": c, "period": p},
                             {"id": "s3", "op": "calc", "expr": expr}], "answer": "s3"}

    def compare2(self):
        """Which of two named companies, and by how much: rank over the pair with k=2."""
        by, w = self.measure(allow_item=True, allow_attr=True)
        pool = self.r.choice([BANKS, CARDS, RETAIL, TECH, INDUSTRIAL, HEALTH])
        a, b = self.r.sample(pool, 2)
        stock = by.get("concept") in STOCKS
        p, pw = self.period(stock=stock) if "attr" not in by else ({"fy": "latest"}, "")
        hi = self.r.choice(["more", "higher", "larger", "the bigger"])
        frames = [f"Which had {hi} {w} {pw}, {{a}} or {{b}}, and by how much?", f"Did {{a}} or {{b}} report {hi} {w} {pw}?",
                  f"Comparing {{a}} and {{b}}, which had {hi} {w} {pw}?", f"Which of {{a}} and {{b}} had {hi} {w} {pw}, and what was the gap?"]
        q = re.sub(r"\s+", " ", self.r.choice(frames).format(a=self.co(a), b=self.co(b))).replace(" ,", ",").replace(" ?", "?")
        return q, self.one({"op": "rank", "over": [a, b], "by": by, "period": p, "order": "desc", "k": 2})

    def share_of(self):
        """A part as a share of the whole: item over concept, or one concept over another."""
        if self.r.random() < 0.65:
            t, item, words = self.item_for()
            whole, ww = ("OperatingIncome", "operating income") if "operating income" in item else ("Revenue", self.r.choice(["revenue", "total revenue", "net sales", "sales"]))
            part = {"id": "s1", "op": "get", "company": t, "item": item, "period": None}
        else:
            t = self.r.choice(self.tickers)
            c, words = self.concept(["ResearchAndDevelopment", "SellingGeneralAdmin", "CapitalExpenditures", "DividendsPaid",
                                     "StockBasedCompensation", "IncomeTax", "InterestExpense"])
            whole, ww = self.r.choice([("Revenue", "revenue"), ("NetIncome", "net income"), ("OperatingCashFlow", "operating cash flow")])
            part = {"id": "s1", "op": "get", "company": t, "concept": c, "period": None}
        p, pw = self.period()
        part["period"] = p
        q = self.r.choice([f"What share of {self.co(t)}'s {pw} {ww} came from {words}?".replace(" in ", " ", 1) if False else
                           f"What share of {self.co(t)}'s {ww} {pw} came from {words}?",
                           f"What was {self.co(t)}'s {words} as a percentage of its {ww} {pw}?",
                           f"How large was {self.co(t)}'s {words} relative to its {ww} {pw}?"])
        return q, {"steps": [part, {"id": "s2", "op": "get", "company": t, "concept": whole, "period": p},
                             {"id": "s3", "op": "calc", "expr": "s1 / s2"}], "answer": "s3"}

    def loss_share(self):
        t, item, words = self.item_for()
        p, pw = self.period()
        q = self.r.choice([f"How big was {self.co(t)}'s {words} loss relative to its total operating income {pw}?",
                           f"How large was the {words} operating loss at {self.co(t)} compared with consolidated operating income {pw}?"])
        return q, {"steps": [{"id": "s1", "op": "get", "company": t, "item": item, "period": p},
                             {"id": "s2", "op": "get", "company": t, "concept": "OperatingIncome", "period": p},
                             {"id": "s3", "op": "calc", "expr": "abs(s1) / s2"}], "answer": "s3"}

    def per_unit(self):
        """A figure divided by a count: per employee, per store, per member, per vehicle."""
        t = self.r.choice(self.tickers)
        c, w = self.concept(["Revenue", "NetIncome", "OperatingIncome"])
        p, pw = self.period()
        if self.r.random() < 0.45:
            q = self.r.choice(["What was {co}'s {w} per employee {pw}?", "How much {w} did {co} generate per employee {pw}?",
                               "Using its headcount, what was {co}'s {w} per worker {pw}?"]).format(co=self.co(t), w=w, pw=pw)
            return q, {"steps": [{"id": "s1", "op": "get", "company": t, "concept": c, "period": p},
                                 {"id": "s2", "op": "attr", "company": t, "attr": "employees"},
                                 {"id": "s3", "op": "calc", "expr": "s1 / s2"}], "answer": "s3"}
        unit, item = self.r.choice([("store", "number of stores"), ("warehouse", "number of warehouses"),
                                    ("paid member", "paid members"), ("vehicle delivered", "vehicles delivered"),
                                    ("restaurant", "number of restaurants"), ("customer", "number of customers")])
        q = self.r.choice([f"How much {w} did {{co}} make per {unit} {pw}?", f"Dividing {{co}}'s {w} by its {item.replace('number of ', '')}, "
                           f"what was the figure per {unit} {pw}?", f"What was {{co}}'s {w} per {unit} {pw}?"]).format(co=self.co(t))
        return q, {"steps": [{"id": "s1", "op": "get", "company": t, "concept": c, "period": p},
                             {"id": "s2", "op": "get", "company": t, "item": item, "period": p},
                             {"id": "s3", "op": "calc", "expr": "s1 / s2"}], "answer": "s3"}

    def item_sum(self):
        if not PARTS:
            return self.get()
        t = self.r.choice(sorted(PARTS))
        labels = [lab for lab, anchor in PARTS[t] if anchor == "Revenue"]
        if len(labels) < 2:
            return self.item_q()
        a, b = self.r.sample(labels, 2)
        p, pw = self.period()
        q = self.r.choice([f"Combined, how much did {self.co(t)}'s {a} and {b} bring in {pw}?",
                           f"What were {self.co(t)}'s {a} and {b} revenues together {pw}?"])
        return q, {"steps": [{"id": "s1", "op": "get", "company": t, "item": f"{a} revenue", "period": p},
                             {"id": "s2", "op": "get", "company": t, "item": f"{b} revenue", "period": p},
                             {"id": "s3", "op": "calc", "expr": "s1 + s2"}], "answer": "s3"}

    def rank(self):
        by, w = self.measure(allow_item=True)
        over, one, many = self.a_set()
        if "item" in by and not isinstance(over, list):
            over, one, many = self.a_set()
            if not isinstance(over, list):
                by, w = {"concept": "Revenue"}, "revenue"
        p, pw = self.period(stock=by.get("concept") in STOCKS)
        order = "desc" if self.r.random() < 0.7 else "asc"
        hi = "highest" if order == "desc" else "lowest"
        if isinstance(over, list) and self.r.random() < 0.5:
            k = len(over)
            q = self.r.choice([f"Rank {many} by {w} {pw}, {'largest' if order == 'desc' else 'smallest'} first.",
                               f"Order {many} by {w} {pw}, {'highest' if order == 'desc' else 'lowest'} first."])
        else:
            k = 1 if self.r.random() < 0.7 else self.r.choice([2, 3, 5])
            if isinstance(over, list):
                q = f"Among {many}, which had the {hi} {w} {pw}?" if k == 1 else f"Among {many}, which {k} had the {hi} {w} {pw}?"
            elif k == 1:
                q = self.r.choice([f"Which {one} had the {hi} {w} {pw}?", f"Which {one} reported the {hi} {w} {pw}?"]
                                  if not self.val else [f"Name the {one} with the {hi} {w} {pw}."])
            else:
                q = self.r.choice([f"Which {k} {many} had the {hi} {w} {pw}?", f"List the {k} {many} with the {hi} {w} {pw}."])
        return q, self.one({"op": "rank", "over": over, "by": by, "period": p, "order": order, "k": k})

    def hop(self):
        by, w = self.measure()
        over, one, many = self.a_set()
        if one is None:
            over, one = "all", "company"
        p, pw = self.period(stock=by.get("concept") in STOCKS)
        order = "desc" if self.r.random() < 0.8 else "asc"
        hi = "highest" if order == "desc" else "lowest"
        first = {"id": "s1", "op": "rank", "over": over, "by": by, "period": p, "order": order, "k": 1}
        if self.r.random() < 0.65:
            attr = self.r.choice(list(HOP_ATTR))
            if attr == "state_of_incorporation":
                q = f"In which state is the {one} with the {hi} {w} {pw} incorporated?"
            else:
                q = f"{self.r.choice(HOP_ATTR[attr])} the {one} with the {hi} {w} {pw}?"
            return q, {"steps": [first, {"id": "s2", "op": "attr", "company": "$s1", "attr": attr}], "answer": "s2"}
        m, mw = self.metric()
        q = f"What was the {mw} {pw} of the {one} with the {hi} {w}?"
        return q, {"steps": [first, {"id": "s2", "op": "metric", "company": "$s1", "metric": m, "period": p}], "answer": "s2"}

    def filt(self):
        over, one, many = self.a_set()
        p, pw = self.period()
        x = self.r.random()
        if x < 0.35:
            m, w = self.metric(ratio_only=True)
            pct = self.r.choice([5, 10, 15, 20, 25, 30, 40])
            op, word = self.r.choice([("gt", "above"), ("lt", "below"), ("ge", "of at least")])
            cond = {"by": {"metric": m}, "period": p, op: pct / 100}
            phrase = f"a {w} {word} {pct} percent {pw}"
        elif x < 0.65:
            c, w = self.concept(["Revenue", "NetIncome", "OperatingCashFlow", "TotalAssets", "ResearchAndDevelopment", "CapitalExpenditures"])
            bn = self.r.choice([5, 10, 20, 50, 100, 200])
            op, word = self.r.choice([("gt", "more than"), ("lt", "less than")])
            cond = {"by": {"concept": c}, "period": p, op: bn * 1_000_000_000}
            phrase = f"{w} of {word} ${bn} billion {pw}"
        else:
            c, w = self.concept(["Revenue", "NetIncome", "OperatingIncome", "EPSDiluted", "OperatingCashFlow"])
            if self.r.random() < 0.5:
                down = self.r.random() < 0.5
                cond = {"by": {"delta": {"concept": c}}, "period": p, ("lt" if down else "gt"): 0}
                phrase = f"{'lower' if down else 'higher'} {w} {pw} than the year before"
            else:
                pct = self.r.choice([5, 10, 20])
                cond = {"by": {"growth": {"concept": c}}, "period": p, "gt": pct / 100}
                phrase = f"{w} growth above {pct} percent {pw}"
        if one is None and self.r.random() < 0.5:
            q = f"Of {many}, how many had {phrase}, and which ones?"
        else:
            q = f"How many {many} had {phrase}?" if self.r.random() < 0.6 else None
        if q:
            return q, {"steps": [{"id": "s1", "op": "filter", "over": over, "where": cond}, {"id": "s2", "op": "count", "of": "$s1"}], "answer": "s2"}
        return f"Which {many} had {phrase}?", self.one({"op": "filter", "over": over, "where": cond})

    def attr_filter(self):
        if self.r.random() < 0.6:
            a = self.r.choice(AUDITORS)
            cond, phrase = {"attr": "auditor", "eq": a}, f"audited by {a}"
        else:
            s = self.r.choice(STATES)
            cond, phrase = {"attr": "state_of_incorporation", "eq": s}, f"incorporated in {s}"
        x = self.r.random()
        if x < 0.35:
            return f"How many companies are {phrase}?", {"steps": [{"id": "s1", "op": "filter", "over": "all", "where": cond},
                                                                    {"id": "s2", "op": "count", "of": "$s1"}], "answer": "s2"}
        by, w = self.measure()
        p, pw = self.period(stock=by.get("concept") in STOCKS)
        if x < 0.7:
            q = f"Among the companies {phrase}, which had the highest {w} {pw}?"
            return q, {"steps": [{"id": "s1", "op": "filter", "over": "all", "where": cond},
                                 {"id": "s2", "op": "rank", "over": "$s1", "by": by, "period": p, "order": "desc", "k": 1}], "answer": "s2"}
        m, mw = self.metric(ratio_only=True)
        pct = self.r.choice([10, 15, 20, 30])
        q = f"How many of the companies {phrase} had a {mw} above {pct} percent {pw}?"
        return q, {"steps": [{"id": "s1", "op": "filter", "over": "all", "where": cond},
                             {"id": "s2", "op": "filter", "over": "$s1", "where": {"by": {"metric": m}, "period": p, "gt": pct / 100}},
                             {"id": "s3", "op": "count", "of": "$s2"}], "answer": "s3"}

    def agg(self):
        by, w = self.measure()
        over, one, many = self.a_set()
        p, pw = self.period(stock=by.get("concept") in STOCKS)
        fn, word = self.r.choice([("sum", "combined"), ("sum", "total"), ("avg", "average"), ("median", "median"),
                                  ("max", "largest"), ("min", "smallest")])
        q = self.r.choice([f"What was the {word} {w} of {'the ' if one else ''}{many} {pw}?",
                           f"Across {'the ' if one else ''}{many}, what was the {word} {w} {pw}?"])
        return q, self.one({"op": "agg", "over": over, "by": by, "period": p, "fn": fn})

    def series(self):
        t = self.r.choice(self.tickers)
        if self.r.random() < 0.6:
            c, w = self.concept(self.plain_concepts())
            by = {"concept": c}
        else:
            m, w = self.metric()
            by = {"metric": m}
        n = self.r.choice([2, 3])
        if self.r.random() < 0.6:
            pick, word = self.r.choice([("max", "peak"), ("max", "reach its highest"), ("min", "hit its lowest")])
            q = f"In which of the last {n} fiscal years did {self.co(t)}'s {w} {word}?"
            return q, self.one({"op": "series", "company": t, "by": by, "last": n, "pick": pick})
        q = self.r.choice([f"Show {self.co(t)}'s {w} for each of the last {n} fiscal years.",
                           f"How did {self.co(t)}'s {w} develop over its last {n} fiscal years?"])
        return q, self.one({"op": "series", "company": t, "by": by, "last": n})

    def members(self):
        t = self.r.choice(self.tickers)
        of = self.r.choice(BREAKDOWNS)
        noun = next((v for k, v in BREAKDOWN_NOUN.items() if k in of), "segments")
        what = of.split(" by ")[0] if " by " in of else of
        p, pw = self.period()
        first = {"id": "s1", "op": "members", "company": t, "of": of, "period": p}
        x = self.r.random()
        if x < 0.35:
            order = "desc" if self.r.random() < 0.75 else "asc"
            q = f"Which of {self.co(t)}'s {noun} had the {'highest' if order == 'desc' else 'lowest'} {what} {pw}?"
            return q, {"steps": [first, {"id": "s2", "op": "rank", "over": "$s1", "by": {"value": True}, "period": p,
                                         "order": order, "k": 1}], "answer": "s2"}
        if x < 0.55:
            q = self.r.choice([f"Which of {self.co(t)}'s {noun} grew {what} fastest {pw}, and how much did it bring in?",
                               f"Which {noun[:-1]} of {self.co(t)} had the fastest {what} growth {pw}, and what was its {what}?"])
            return q, {"steps": [first, {"id": "s2", "op": "rank", "over": "$s1", "by": {"growth": {"value": True}}, "period": p,
                                         "order": "desc", "k": 1},
                                 {"id": "s3", "op": "get", "company": "$s2", "period": p}], "answer": "s3"}
        if x < 0.75:
            q = self.r.choice([f"Which of {self.co(t)}'s {noun} posted a loss {pw}, and how big was it?",
                               f"Did any of {self.co(t)}'s {noun} lose money {pw}?"])
            return q, {"steps": [first, {"id": "s2", "op": "filter", "over": "$s1", "where": {"by": {"value": True}, "period": p, "lt": 0}}],
                       "answer": "s2"}
        if x < 0.88:
            q = f"How many of {self.co(t)}'s {noun} grew {what} {pw}?"
            return q, {"steps": [first, {"id": "s2", "op": "filter", "over": "$s1", "where": {"by": {"growth": {"value": True}},
                                                                                                "period": p, "gt": 0}},
                                 {"id": "s3", "op": "count", "of": "$s2"}], "answer": "s3"}
        q = f"Rank {self.co(t)}'s {noun} by {what} {pw}."
        return q, {"steps": [first, {"id": "s2", "op": "rank", "over": "$s1", "by": {"value": True}, "period": p,
                                     "order": "desc", "k": 10}], "answer": "s2"}

    def attr(self):
        a = self.r.choice(list(ATTR_Q))
        t = self.r.choice(self.tickers)
        return self.pick(ATTR_Q[a]).format(co=self.co(t)), self.one({"op": "attr", "company": t, "attr": a})

    def brand_attr(self):
        brand, t = self.r.choice([("Gatorade", "PEP"), ("Instagram", "META"), ("YouTube", "GOOGL"), ("WhatsApp", "META"),
                                  ("Sprite", "KO"), ("Tide", "PG"), ("Jordan", "NKE"), ("AWS", "AMZN"), ("Azure", "MSFT"),
                                  ("Sam's Club", "WMT"), ("Optum", "UNH"), ("Keytruda", "MRK"), ("Chevrolet", "GM"),
                                  ("Lincoln", "F"), ("Slack", "CRM"), ("Photoshop", "ADBE"), ("Webex", "CSCO"),
                                  ("Mobil", "XOM"), ("Texaco", "CVX"), ("Kirkland Signature", "COST"), ("Lay's", "PEP")])
        a, frame = self.r.choice([("auditor", "Who audits the company that owns {b}?"), ("state_of_incorporation",
                                  "In which state is the maker of {b} incorporated?"), ("ceo", "Who runs the company behind {b}?"),
                                  ("hq_city", "In which city is the company that owns {b} headquartered?")])
        return frame.format(b=brand), self.one({"op": "attr", "company": t, "attr": a})

    def restated(self):
        t = self.r.choice(self.tickers)
        y = self.r.choice([2024, 2025])
        if self.r.random() < 0.6:
            c, w = self.concept(self.plain_concepts())
            measure = {"concept": c}
        else:
            t, item, w = self.item_for(t)
            measure = {"item": item}
        q = self.r.choice([f"Did {self.co(t)} restate its fiscal {y} {w} in the following year's filing, and by how much?",
                           f"What did {self.co(t)} first report as {w} for fiscal {y}, what does the next annual report show, and why the difference?",
                           f"Did the fiscal {y} {w} figure of {self.co(t)} change in the later filing, and if so why?",
                           f"How did {self.co(t)}'s reported {w} for fiscal {y} change between the original filing and the next one?",
                           f"{self.co(t)}'s fiscal {y + 1} annual report shows a different fiscal {y} {w} than the fiscal {y} report did. By how much was it changed?"])
        return q, self.one({"op": "restated", "company": t, **measure, "period": {"fy": y}})

    def cannot(self):
        if self.r.random() < 0.5:
            name = self.r.choice(NOT_LISTED)
            c, w = self.concept(self.plain_concepts())
            p, pw = self.period(stock=c in STOCKS)
            q = self.r.choice([f"What was {name}'s {w} {pw}?", f"How did {name}'s {w} compare with {self.co(self.r.choice(self.tickers))}'s {pw}?"])
            return q, {"steps": [], "answer": None, "cannot": f"{name} is not among the listed companies"}
        frame, why = self.r.choice(CANNOT)
        t = self.r.choice(self.tickers)
        return frame.format(co=self.co(t), yr=self.r.choice([2024, 2025])), {"steps": [], "answer": None, "cannot": why}

    def rank_attr(self):
        over, one, many = self.a_set()
        attr, word = self.r.choice([("employees", "employees"), ("employees", "people on its payroll"),
                                    ("shares_outstanding", "shares outstanding")])
        order = "desc" if self.r.random() < 0.75 else "asc"
        if isinstance(over, list):
            k = len(over)
            q = f"Rank {many} by {word}, {'most' if order == 'desc' else 'fewest'} first."
        else:
            k = 1
            q = f"Which {one} has the {'most' if order == 'desc' else 'fewest'} {word}?"
        return q, self.one({"op": "rank", "over": over, "by": {"attr": attr}, "period": {"fy": "latest"}, "order": order, "k": k})

    def sample(self):
        makers = [(self.get, 13), (self.first_reported, 2), (self.metric_q, 7), (self.item_q, 12), (self.change, 7),
                  (self.compare_calc, 5), (self.compare2, 8), (self.share_of, 6), (self.loss_share, 2), (self.per_unit, 4),
                  (self.item_sum, 2), (self.rank, 9), (self.hop, 5), (self.filt, 6), (self.attr_filter, 3), (self.agg, 4),
                  (self.series, 3), (self.members, 8), (self.attr, 3), (self.brand_attr, 1), (self.restated, 4),
                  (self.cannot, 3), (self.rank_attr, 2)]
        fn = self.r.choices([m for m, _ in makers], weights=[w for _, w in makers])[0]
        return fn()


def system_prompt() -> str:
    """The short prompt the local planner is trained and run with.

    A fine-tuned model learns the rules from the examples, so the prompt names the task, the
    operations and the vocabulary, and nothing else. It is a third the length of the hosted
    planner's prompt, which keeps training and CPU inference fast.
    """
    lines = ["Translate the question into a TieOut query plan. Output JSON only: "
             '{"steps": [...], "answer": "<step id>"} or {"steps": [], "answer": null, "cannot": "<why>"}.',
             "OPS: get, metric, calc, change, rank, filter, count, agg, series, members, attr, restated.",
             "MEASURES: concept, metric, item (a line outside the three statements, in the filing's words), attr, "
             "growth, delta, ratio, value (of a line).",
             "COMPANIES: " + ", ".join(f"{t} ({ALIASES[t][0]})" for t in ALIASES),
             "CLUSTERS: " + ", ".join(CLUSTERS),
             "CONCEPTS: " + ", ".join(concept_defs()),
             "METRICS: " + ", ".join(metric_defs()),
             "ATTRIBUTES: auditor, auditor_since, auditor_location, state_of_incorporation, fiscal_year_end, hq_city, hq_state, "
             "exchange, trading_symbol, employees, shares_outstanding, public_float_usd, ceo, cfo"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=9000)
    ap.add_argument("--val", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    g = Graph.load()
    OUT.mkdir(parents=True, exist_ok=True)
    seen_train: set = set()
    for name, n, val, seed in (("train", a.train, False, a.seed), ("val", a.val, True, a.seed + 1)):
        gen, seen, rows, rejected = Gen(seed, val), set(), [], 0
        while len(rows) < n:
            q, plan = gen.sample()
            q = _fix_possessives(re.sub(r"\s+", " ", q).strip())
            if q in seen or (val and q in seen_train):
                continue
            if plan.get("steps") and validate(plan, g):
                rejected += 1
                continue
            seen.add(q)
            rows.append({"question": q, "plan": plan})
        if not val:
            seen_train = seen
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        shapes = {}
        for r in rows:
            key = "+".join(s["op"] for s in r["plan"]["steps"]) or "cannot"
            shapes[key] = shapes.get(key, 0) + 1
        print(name, len(rows), "rejected by validator:", rejected, "shapes:", len(shapes))
    (ROOT / "finetune" / "system_prompt.v2.txt").write_text(system_prompt(), encoding="utf-8")
    print("system prompt:", len(system_prompt().split()), "words; item labels from raw layer for", len(PARTS), "companies")


if __name__ == "__main__":
    main()
