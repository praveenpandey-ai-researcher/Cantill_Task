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
