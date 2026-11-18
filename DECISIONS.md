# DECISIONS

Data forensics and every assumption made. Written as the work happens.

## 0. Setup
- Repo root is `Cantill_Task/`. Task data stays outside the repo (`../company_A`, `../company_B`)
  and `.gitignore` blocks all `*.csv` except `outputs/` and `submission/`.
- Config file paths are relative to `data_dir`; `--data-dir` overrides it so the
  blind-test files (A2, B2) run with the unchanged A and B configs.
- Dependency versions are pinned. On the build PC, Windows Smart App Control blocked
  the compiled files of the newest pandas, scikit-learn and tokenizers wheels
  ("An Application Control policy has blocked this file"). Slightly older, widely used
  releases load fine, so `requirements.txt` pins those (pandas 2.2.3, scikit-learn
  1.5.2, torch 2.6.0 CPU, tokenizers 0.21.0, sentence-transformers 3.3.1).

## 1. Data forensics (stage 1)

All numbers below come from `python analysis/explore_data.py`, which prints aggregates only.

### Company A: ERP goods receipts (37,029 lines)
| Finding | Number | Decision |
|---|---|---|
| `posting_date` blank | 24,370 (66%) | Dates are not reliable for matching. Date windows are a tie-breaker only, never a requirement. Blank date is logged, not an error. |
| `vendor_id` blank | 24,370, exactly the same rows | These lines have no vendor-master record (see supplier keys below). |
| Negative `value_zar` | 434 | Goods returns or reversals. **Kept as negative spend**, so receipt and return net out. |
| `document_currency` EUR/CHF/MZN | 16 lines | The amount column is `value_zar` (already ZAR). **No conversion.** |
| `po_number` | always 10 digits (`45100xxxxx`) | ERP PO is the clean reference that tool POs are normalised to. |
| Material number blank | 15,816 | Services / free text. Item de-dup has to fall back on text. |
| ERP exact duplicate rows | 0 | No count-once issue inside the ERP. |

**Golden supplier keys (labels_A).** 193 supplier names map to 70 keys, but this is not
mostly spelling variants. Only 2 keys join spellings (`YARFIELD` / `YARF IELD`, double spaces).
One key holds **20,891 lines and 122 different names, and 99.2% of its lines have a blank
`vendor_id`**. So the client's rule is: suppliers without a vendor-master record go into one
"unmanaged" bucket. Assumption: A2 is scored with the same golden keys, so the supplier
de-dup reproduces this through a config switch (blank vendor id -> one shared key). That is
a company rule, so it lives in the config and is never hard-coded.

**Golden item keys.** `item_key` is blank exactly when `material_number` is blank (15,816
lines), so the client only keys catalogued materials. One material number never maps to two
keys. 20 keys join several material numbers, and one key holds 192 materials (Catalogue / Free
Text). Texts for the same key differ by decimal comma vs dot (`3,5M` vs `3.5M`) and spacing,
so normalising text must treat `,` and `.` inside numbers the same.

**Taxonomy A:** 5 L1 / 24 L2 / 40 L3. In 77.6% of labelled lines L3 repeats L2's name, so many
leaves sit effectively at L2. Every labelled path exists in the taxonomy file.

### Company A: invoice tool (28,520 lines)
| Finding | Number | Decision |
|---|---|---|
| Status | Paid 19,837 / Approved 8,180 / Cancelled 188 / On Hold 172 / Rejected 143 | Paid, Approved -> count. Cancelled -> `CANCELLED`. On Hold, Rejected -> `NOT_APPROVED`. Mapping lives in config. |
| `vat_rate` formats | `15` 6,920, `0.15` 6,826, `15%` 6,652, blank 6,542, `0` 1,580 | Parse all to a fraction (>1 means percent). Blank -> 15% (brief). |
| VAT `0` means really 0 | 94.6% of PO-matched `0` lines equal the ERP value **as-is**, 0.8% after /1.15 | `0` is a real zero rate: net = gross. |
| VAT blank means 15% | 96.4% match ERP after /1.15 | Confirms the brief's 15% default. |
| `invoice_date` blank | 18,660 (65%) | As in ERP: dates are a weak signal. |
| Date formats | `dd/mm/yyyy` 3,362, ISO 3,283, `dd-Mon-yy` 3,215 | Slash dates are **day-first**: part 1 > 12 in 2,050 rows, part 2 never > 12. |
| Gross amount | thousands separators in ~30% (`99,999.99`) | Strip `,` thousands separators before parsing. |
| PO reference | bare 5,895 / `PO#` 5,820 / `PO ` 5,807 / `po/line` 5,799 / blank 5,199 | Normalise = extract the 10-digit number. |
| Invoice number | 5 styles: `IN0123456`, `TAX INV 123456`, `SI-2023-123456`, `INV/2023/123456`, bare | Normalise = last digit group, leading zeros stripped. The year inside the number is not reliable (same invoice seen as `INV/2023/x` and `INV/2024/x`). |
| Twice-exported invoices | 424 lines repeat an earlier line on invoice core + supplier + text + net | The copies often differ in invoice style, PO style and even date, so they are matched on normalised fields, not raw ones. Later copy -> `DUPLICATE`. |

**Cross-system overlap (tool vs ERP).**
| Tool lines | Count |
|---|---|
| PO found in ERP and net amount matches to the cent | 16,743 |
| PO found in ERP, amount different | 877 (partial or combined invoices; same item text, often OCR-garbled `B1T` / `BIT`) |
| PO not in ERP | 5,701 |
| No PO | 5,199 |

- 72% of no-PO lines match an ERP line exactly on supplier + net amount, so the
  supplier + amount fallback is worth having.
- 38% of "PO not in ERP" lines also match supplier + net. A digit-typo explanation can't be
  tested: ERP POs are so dense that any 10-digit number is one edit away from a real PO.
  Stage 3 decides this with one-to-one matching, so that an ERP line already claimed by another
  tool line can't be claimed twice.
- Supplier names: 401 normalised tool names; 85.8% of tool lines use a name that exists
  exactly in the ERP.
- Tool texts with no information: `NO OBJECT` / `N0 OBJECT` / `NOOBJECT` (~690 lines),
  `MECHANICAL MAINTENAN`, `HAND TOOLS`. These need supplier-based fallback in classification.
- No tax, royalty, intercompany or finance text found in A (keyword scan of ERP and tool).
  All ERP lines carry a category label, so A's exclusions are expected to come only from tool
  status and duplicates.

### Company B: invoice lines (41,028 lines, 8 ERPs)
| Finding | Number | Decision |
|---|---|---|
| **No status column** | n/a | Every line is a posted AP invoice. There is nothing to drive CANCELLED / NOT_APPROVED, and the seed labels contain neither code. Assumption: B has no status exclusions. A blank `paid_date` (10,322) means posted but not yet paid, which still counts as approved. |
| Rate direction | given `currency_rate` / table rate: median 1.000 (1st-99th pct 0.95-1.04) | Confirms "units per USD": **USD = amount / rate**. |
| USD lines carrying a rate | 1,279 (rate ~17.4 = MXN rate) | Trap. Brief: USD lines are already USD. **Ignore the rate when currency is USD.** |
| Non-USD lines without a rate | 3,343 (CAD 1,830, MMK 1,077, THB 270, ...) | Use `fx_rates_usd.csv` for the invoice month. If that month is missing, use the posting month, then the nearest month in the table, and log it. |
| Currencies | 25, all present in the FX table (including `RCN`) | |
| Implausible invoice dates | 2018-2022: 76, 2027: 2; posting dates are all 2023-12/2024 | Flag dates that are > 2 years before or > 1 year after the posting date as suspicious and use the posting date for FX. The original value is kept. |
| Amount blank | 3 | Spend 0 and logged as unparsable; never crash. |
| Negative amounts | 833 | Credit notes / reversals (311 +x / -x pairs). **Kept as negative spend.** The seed labels include 20 negatives. |
| Zero amounts | 918 | Kept (spend 0), classified as normal. |
| Exact duplicate rows | 613 (ERP-B3 512, B8 69, B7 32) | Every field except `line_id` identical. Later copy -> `DUPLICATE`. Evidence: a random sample of 3,000 would contain ~90 rows from these groups, but the seed has only 2, so the analyst left them out on purpose. |
| Blank descriptions | 16,695 (41%) are empty or a single space (SAP regions write `" "`); 4,885 more are numbers only | Classification must use GL account, material number, vendor, PO type and plant when text is empty. |
| Text | Many languages (EN, ES, PT, FR, ...); JDE regions append `" - "` | Strip a trailing `-` and whitespace. Multilingual embedding model. |

**What B's seed labels say about NON_PROCUREMENT (450 of 3,000 seed lines).**
- 364 are intercompany. Every seed line whose GL description starts with `IC<nn>`, `IC Payables`,
  `I/C Payable` or `Intercompany` is excluded, and none of them is included. Rule: those GL
  descriptions -> `NON_PROCUREMENT` (2,041 lines in the full file).
- The other 86 are taxes by text: `State & Local Taxes` 49, `Sales Tax Expense` 22,
  `GST Input Tax` 10, `Sales Tax` 5. The same vendors also bill freight, postage and supplies
  that are included, so **tax is detected from the text, not the vendor**.
- Keyword traps: whole-word, phrase-level keywords are needed.
  - Bought services that merely mention tax stay in: `TAX SERVICES`, `BLANKET PO FOR TAX WORK`,
    `ASSISTANCE WITH PROPERTY TAX` are advisory services from a third party.
  - Naive keywords would wrongly hit `FINE GRIT`, `HEAVY DUTY`, `BANKERS BOX`, and `BANCO`
    (Spanish "bench").
  - `IMPORT DUTY` / `CUSTOMS DUTY` are taxes paid to customs, but taxonomy B has a leaf
    `Customs Clearance & Duties`. Assumption: duty lines **stay in** and are classified there,
    because the client built a category for them.
- Partnership fees (4 lines, e.g. "PARTNERSHIP FEES 2024-2025"): sponsorship or membership,
  bought from a third party. Assumption: kept as spend. "Partner billings" in the brief means
  money passed to business partners, which these are not.

**Taxonomy B:** 9 L1 / 25 L2 / 40 L3 (L1 `Facilities` appears twice in the file, not
contiguously, so the tree loader must group by name, not by row order).

## 2. Cleaning: amounts, VAT, FX, dates (stage 2)

Code: `spendpipe/clean.py`. Tests: `tests/test_clean.py`, `tests/test_ingest_clean.py`.

**Amounts.** Every cell is read as text, then parsed by one function that handles
`1,234.56`, `1.234,56`, `1 234,56`, `(500)`, SAP trailing minus `500-` and currency symbols.
With both `,` and `.` present, the last one is the decimal mark. A lone `,` is a decimal mark
only when 1-2 digits follow it. A config can force the convention per source. Unparsable
amounts give spend 0 plus the flag `AMOUNT_UNPARSABLE` (B: 3 lines).

**VAT (A tool).** Rate parsed to a fraction (`15`, `15%`, `0.15` -> 0.15). Blank -> 15%,
`0` -> 0%, as stage 1 showed. net = gross / (1 + rate). Result: 26,940 lines at 15%,
1,580 at 0%.

**Currency.**
- A source's `currency.fixed` means the amount column is always in that currency. A's
  `value_zar` is ZAR even on the 16 lines whose `document_currency` says EUR/CHF/MZN; the
  original value is kept as `currency_doc`.
- Order per line:
  1. Same as the reporting currency -> rate 1, any given rate ignored (B: 1,279 USD lines with
     a stray rate).
  2. Given rate.
  3. FX table for the invoice month.
  4. Nearest month in the table.
- B result: given 19,667 / same-currency 17,959 / table 3,338 / nearest month 5.
- **Wrong given rates.** 59 non-USD lines (55 EUR, 4 CHF) carry a rate of about 17.6, an
  MXN-sized rate, 19x away from the table. Rule (config `fx.max_rate_deviation: 5`): a given
  rate more than 5x away from the table rate for that currency and month is rejected and the
  table rate is used, flagged `FX_GIVEN_RATE_REJECTED`. 5x leaves room for real devaluations:
  2 ARS lines at ~2.3x off around the Dec-2023 devaluation are kept.
- **Tiny-currency lines.** PYG (median amount 0.08), CLP (0.77) and MMK (151) give tiny USD
  values. This might be an SAP storage quirk for zero-decimal currencies, but nothing in the
  brief says so. **Assumption:** apply the brief's rule literally (amount / rate) and don't
  re-scale. These lines total under 1,000 USD, so the effect on totals is negligible.

**Dates.**
- Parser order:
  1. Explicit formats from the config.
  2. ISO.
  3. `YYYYMMDD`.
  4. Excel serial numbers.
  5. A flexible day-first parser that also understands Spanish, Portuguese, French, German
     and Italian month names.
- Each unique value is parsed once.
- Flags:
  - `blank` (A ERP 24,370; A tool 18,660).
  - `unparsable`, which covers impossible calendar dates such as `2023-02-30`.
  - `implausible`: outside the config window `dates.min`/`max` (2019-01-01 to 2025-12-31), or
    more than `max_gap_days` (B: 730) from the posting date. B: 42 lines.
- For FX, an implausible invoice date falls back to the posting month.
- Bad dates never drop a line: the original value is kept, the line gets a flag, and the
  line stays in the output.

**Text.** Trim, collapse whitespace, strip trailing ` -` (JDE). Single-space SAP
descriptions become blank: B 16,695 `DESC_BLANK`.

**Runtime so far:** ingest + clean takes 1.2 s for A (65,549 rows) and 0.8 s for B.

## 3. Count once: matching A's two systems (stage 3)

Code: `spendpipe/match.py`, `spendpipe/normalise.py`, status rules in `spendpipe/exclude.py`.
Evidence: `python analysis/eval_matching.py` (aggregates only). Tests: `tests/test_match.py`.

There are no labels saying which tool line duplicates which ERP line, so every rule
below was chosen from measured evidence, and the evaluation script re-checks it.

**Order of work**
1. **Status first.** Paid/Approved count; Cancelled -> `CANCELLED` (188); On Hold/Rejected ->
   `NOT_APPROVED` (315). Unknown status -> `NOT_APPROVED` (brief: only approved or paid
   count). Non-counting tool lines never take part in matching: a cancelled invoice must not
   claim an ERP receipt and leave the real invoice looking like extra spend. When a line is
   both cancelled and a duplicate, the reason reported is `CANCELLED` (the record's own state
   wins). Precedence: CANCELLED > NOT_APPROVED > DUPLICATE > NON_PROCUREMENT.
2. **Cross-system passes**, strongest evidence first. Each pass is one-to-one (greedy,
   best text similarity first), so an ERP line can be claimed only once. Keys are normalised:
   PO = the 10-digit number (`PO#`, `PO `, `/10.0` removed); PO line from ERP `po_item` or the
   part after `/`; amount = net of VAT, within max(0.05 ZAR, 0.1%).

   | Pass | Rule | Tool lines |
   |---|---|---|
   | `po_line_amount` | same PO + PO line + net amount | 4,162 |
   | `po_amount` | same PO + net amount | 12,516 |
   | `po_text` | same PO, item text similarity >= 75, amount differs (partial / combined invoice) | 452 |
   | `supplier_amount` | **no PO**: supplier similarity >= 85 + same net amount + same quantity + dates within 31 days (if both known) | 3,881 |
   | `po_text_any` | same PO + similar text, ERP line already claimed (many-to-one) | 387 |

3. **Re-exported invoices** among the tool lines still unmatched: same invoice-number core +
   supplier + net amount + normalised text -> later copy `DUPLICATE` (289). This runs **after**
   matching on purpose. In the first version it ran before, and for 441 of the 739 "copies"
   that had a PO a free identical ERP receipt existed. Those were two genuine identical lines
   (the ERP has such pairs, e.g. two identical fan deliveries), and each should claim its own
   receipt. For the same reason the exact-duplicate-row rule is not applied to the secondary
   source. 90% of the remaining re-export copies differ from the first copy in raw PO or date
   format, which shows they came from a second export.

**Why the supplier + amount fallback is only for tool lines with NO PO.**
- **PO typo test.** For tool lines whose PO is not in the ERP, the first version also matched
  1,666 of them on supplier + amount. If their POs were typos, the matched ERP PO would be 1-2
  edits away. The edit-distance distribution was the same as against a *random* ERP PO
  (median 5). So these were coincidences: the same supplier sells the same item at the same
  price again and again.
- **Signals compared across match types.**

  | Match type | Date gap (median / p90) | Same quantity | Text >= 90 |
  |---|---|---|---|
  | PO + amount (true duplicates) | 11 / 22 days | 100% | 92% |
  | Supplier + amount, no PO | 9 / 22 days | 97% | 90% |
  | Supplier + amount, tool PO not in ERP | 22 / 53 days | 85% | 76% |

- **Decision:** a tool PO that is not in the ERP marks **tool-only spend**. No-PO lines may use
  the fallback, which also requires the same quantity and a 31-day window when both dates
  exist.

**Why `po_text_any` (many-to-one).** After the one-to-one passes, 413 counting tool lines had a
PO that exists in the ERP but no match. For 379 of them **every** ERP line of that PO was
already claimed, and their text matched the PO's item (median similarity 97). These are extra
split or repeated invoices for goods the ERP already records. Count-once says the ERP receipt
is the purchase, so they are `DUPLICATE`. Only 50 tool lines with an ERP PO stay tool-only
(their text doesn't match any line of that PO).

**Checks** (`analysis/eval_matching.py`):
- For every pass, 100% of matched tool lines have a supplier name that maps (through labels_A
  on ERP names) to the same golden supplier key as the matched ERP line.
- `po_text` pairs differ in quantity by design (1% same quantity), which confirms they are
  partial invoices.

**Result for A**
- Tool lines: 21,687 `DUPLICATE`, 6,330 tool-only included, 188 `CANCELLED`, 315 `NOT_APPROVED`.
- Of the tool-only lines, 5,107 have a PO not in the ERP, 1,173 have no PO, and 50 have an
  ERP PO with unrelated text.
- Counted spend: ERP ZAR 2,553.2 m + tool-only ZAR 442.6 m = **ZAR 2,995.8 m**.
- Matching takes 2.5 s.

**B:** a single system, so there is no cross-matching; 613 exact duplicate rows -> `DUPLICATE`.

**Normalisation fixes found by tests.** Dotted legal forms such as `S.A. de C.V.` are collapsed
before suffix stripping, so they strip properly.

## 4. Exclusions: NON_PROCUREMENT and precedence (stage 4)

Code: `spendpipe/exclude.py`. Shared lexicon: `configs/lexicons/non_procurement.yaml`.
Evidence: `python analysis/eval_exclusions.py --config configs/company_B.yaml`. Tests: `tests/test_exclude.py`.

**How a line becomes NON_PROCUREMENT.** The first rule that fires is recorded in `np_rule`,
so every exclusion can be explained:
1. GL account codes listed in the config (`accounts`).
2. GL account-description regex (`account_desc_patterns`): B's intercompany accounts.
3. Supplier ids or supplier-name patterns (tax authorities and similar, plus config extras).
4. The shared lexicon on the description. Each group (tax, intercompany, finance, royalty,
   revenue/partner, payroll, government/fines) has phrase patterns plus `unless` patterns.
   Phrases and word boundaries are used, never bare substrings. The lexicon covers EN, ES, PT,
   FR, DE, TR and RU terms.

The lexicon is a config file shared by all companies, so C only needs to reference it; any
company-specific accounts or suppliers go in that company's config.

**B: intercompany.**
- Rule: GL description matches `^IC\d`, `^IC\s`, `^I/C\b` or `^Intercompany\b`.
  2,041 lines.
- Intercompany vendors post 99-100% of their lines to these accounts (75 of 76 vendors). The
  only exception line was *included* by the analyst, so no vendor-level rule is used.

**B: taxes by text**: 450 lines (`STATE & LOCAL TAXES`, `SALES TAX EXPENSE`,
`GST INPUT TAX`, `ACCRUED SALES & USE TAX`, Turkish `KDV %20`, ...).

**Score against the 3,000 seed labels:** NON_PROCUREMENT precision **1.000**, recall
**1.000** (450 / 450, no included line excluded).

**False positives found on the full file and fixed** (the seed has none of these):
- `CISCO ISR 1100 ROUTER`: `ISR` (a Mexican tax code) removed from the lexicon.
- `CUSTOMS CLEARENCE ... 2024 VAT`: `customs` / `clear` added to the tax `unless` list.

Remaining included lines that mention tax-like words, each kept on purpose:
- Bought services: `TAX SERVICES`, `TAX CONSULTANT`, `PAYROLL SERVICES`, `TAX WORK`,
  `ASSISTANCE WITH PROPERTY TAX`, `R&D TAX C(redit)` (advisory), `D&B FINANCE ANALYTICS`.
- `IMPORT DUTY` / `CUSTOMS DUTY` (5 lines): B's tree has `Customs Clearance & Duties`, so
  duty stays as spend in that leaf (assumption from stage 1).
- `PARTNERSHIP FEES`, `WORLD ECONOMIC FORUM PARTNERSHIP`, `SAP & OMP PARTNERS`:
  memberships or IT partners paid as suppliers. These are not "partner billings".
- `HEAVY DUTY ...`, `MEDIUM DUTY ...`: product descriptions.

**A:** zero NON_PROCUREMENT hits, consistent with stage 1 (every ERP line is a labelled
category, no tax or intercompany text).

**Precedence when several reasons apply:** CANCELLED > NOT_APPROVED > NON_PROCUREMENT >
DUPLICATE.
- The record's own nature comes first.
- DUPLICATE means "already counted in another record". A copy of an intercompany or tax line
  is not "already counted", because the original isn't counted either, so it stays
  NON_PROCUREMENT. This affects 9 lines in B.

**Result**
| | Included | DUPLICATE | NON_PROCUREMENT | NOT_APPROVED | CANCELLED | Included spend |
|---|---|---|---|---|---|---|
| A | 43,359 | 21,687 | 0 | 315 | 188 | ZAR 2,995.8 m |
| B | 37,933 | 604 | 2,491 | 0 | 0 | USD 328.6 m |

## 5. Supplier and item de-duplication (stage 5)

Code: `spendpipe/dedup.py`. Evidence and threshold sweeps: `python analysis/eval_dedup.py`.
Tests: `tests/test_dedup.py`.

**Metric.** Pairwise precision / recall over lines: of all line pairs we put in the same
group, how many the client also groups (precision), and of the client's pairs, how many we
group (recall). This is measured on A's 37,029 labelled ERP lines.

### Suppliers
Method:
1. Normalise names: upper-case, strip accents, punctuation and legal suffixes (`(PTY) LTD`,
   `CC`, `S.A. de C.V.`, `GmbH`, ...), and squash spaces (`YARF IELD` = `YARFIELD`).
2. Build a graph that links each name to its vendor id, plus fuzzy edges between names in
   the same 3-letter block (rapidfuzz ratio >= 92).
3. Connected components are the suppliers.

**Client rule (A config, `unmanaged_bucket`).** 96% of all golden supplier pairs come from
one golden key that holds 122 differently named suppliers, and 99.2% of its lines have no
`vendor_id`. So a supplier whose ERP lines mostly lack a vendor id goes into one shared key,
`UNMANAGED`.
- Name clustering only: P 1.000 / R 0.100 (each unmanaged supplier stays separate).
- Bucket by line (blank id -> bucket): P 0.729 / R 0.971.
- **Bucket by supplier (majority of its lines blank): P 1.000 / R 1.000.**

This is a client convention, so it lives in A's config. C and B don't get it unless their
config turns it on.

**Generic path (for C), tested on realistic noise.** The tool's as-captured supplier names
were clustered together with ERP names, using the supplier of the ERP line each tool line
matched in stage 3 as truth (191 names, 58k lines). Exact normalised names already give
P = R = 1.000, and every fuzzy threshold from 85 to 95 stays at 1.000 (80 drops to P 0.9997).
92 is used for a safety margin.

**B:** no supplier names exist, only vendor ids, so supplier_key = vendor id (3,995 keys).
Different vendor ids for one real supplier can't be detected without names. Assumption:
each vendor id is one supplier.

DUPLICATE lines take the supplier key of the record they duplicate. All tool lines in A end
up on supplier keys that also exist on ERP lines.

### Items
Order:
1. Material code.
2. Config bucket rules.
3. Inherit from the duplicated record.
4. Text -> code mapping.
5. Text clustering.
6. Vendor + GL fallback.

**Client rules (A config).**
- `item_key` is blank exactly when there is no material number (15,816 lines), so
  `no_code: blank`.
- 66% of keyed golden pairs come from one key: Catalogue / Free Text materials with a blank
  material type (192 materials, 2,659 lines). This becomes bucket rule
  `NONCATALOGUE_MATERIAL` (only for lines with a code).
- Material code alone: P 1.000 / R 0.474 on keyed lines. **With the bucket rule: P 0.996 /
  R 0.995 (blank counted as a group: 0.9998 / 0.9998).**
- Merging material numbers that share the same text *hurts* (P 0.789) because the client
  gives identical texts different keys. Not done.

**Text -> code for lines without a code.**
- The tool never captures material numbers. A code-less line whose text matches a coded
  item's text takes that item's key: TF-IDF character 2-4-gram cosine >= 0.6 with identical
  numbers, or >= 0.8 regardless of numbers, because OCR turns letters into digits (`B1T`,
  `CYL1NDER`).
- Tested on 12,500 tool lines matched to keyed ERP lines (truth known): 90% get a key, and
  98.9% of those keys are correct.
- In A this applies only to TOOL (config `text_to_code.sources`). The first version also gave
  keys to ERP service lines, which dropped recall from 0.9998 to 0.836, because in the ERP a
  missing material number means a service the client leaves unkeyed. Tool lines that
  duplicate an ERP line inherit its key even when it is blank.
- Result: 47% of A's tool-only invoices get a catalogued material's key.

**Generic text clustering (for C / B lines without a code).**
- Method: TF-IDF character n-grams, cosine >= 0.85, connected components, and **two texts
  may only join if their numbers are identical**.
- Without that rule, size variants chain together (`DETONATOR 3.5M,7` ~ `3.5M,8`): P 0.70 at
  cosine 0.9.
- With it, P 0.970 / R 0.995 (codes hidden, A's keyed lines). Exact text gives the same
  0.970 ceiling, where the client keys identical texts differently.

**No code and no text** (B: 4,195 lines, single-space descriptions): the key is vendor + GL
account (`unknown:V500123|GL40125`). Same vendor and account is most likely the same kind of
purchase, which is better than one blank group.

**Runtime:** de-dup takes 2.8 s for A and 4.7 s for B.
