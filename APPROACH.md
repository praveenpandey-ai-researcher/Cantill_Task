# APPROACH

**Pipeline** (`python -m spendpipe run --config <company>.yaml --out <file>`). Everything
company-specific is in a YAML config: files, column mapping, formats, status words, exclusion
rules, client conventions, model settings. The code only knows canonical fields (`amount`,
`date`, `supplier`, ...) and reads the category tree from its file.

`ingest -> clean -> status -> count once -> non-procurement -> de-dup -> classify -> output + report`

1. **Ingest and clean.** All cells are read as text.
   - Amounts parse any format (`1.234,56`, `(500)`, `500-`).
   - VAT is removed from gross amounts (tool: rate as shown, 15% if blank, 0% kept as 0).
   - FX order: same currency -> 1, ignoring stray rates; then the given rate (divide), unless it
     is more than 5x off the table; then the monthly table for the invoice month; then the
     nearest month.
   - Dates are parsed with explicit, ISO, Excel-serial and multilingual parsers, then flagged
     blank / unparsable / implausible.
   - Nothing crashes: bad values become flags in the report.
2. **Count once (A).** Status is applied first, so a cancelled invoice can't claim an ERP line.
   - One-to-one passes, strongest evidence first: PO + line + net amount -> PO + amount ->
     PO + similar text (partial invoices) -> supplier + amount + quantity (+ date window)
     for tool lines **without** a PO.
   - Then many-to-one PO + text for extra invoices against fully-invoiced POs, then re-exported
     invoices (normalised invoice number + supplier + amount + text).
   - Each rule was chosen from evidence. For example, the PO edit-distance test showed that
     tool POs missing from the ERP are tool-only spend, not typos.
3. **Exclusions.** CANCELLED / NOT_APPROVED from status words (config + multilingual
   defaults). NON_PROCUREMENT from GL accounts/descriptions (B's intercompany), supplier
   patterns, and a shared multilingual phrase lexicon with "unless" guards ("TAX SERVICES" is
   spend). B seed: precision 1.000, recall 1.000 on 450 exclusions. Precedence: CANCELLED >
   NOT_APPROVED > NON_PROCUREMENT > DUPLICATE.
4. **De-dup.**
   - Suppliers: normalised names (legal suffixes, spacing), links through vendor ids,
     rapidfuzz edges inside blocks, connected components.
   - Items: material code -> text-to-code match (OCR tolerant) -> TF-IDF text clusters where
     numbers must match.
   - A's labels showed two client conventions (an "unmanaged supplier" bucket, a
     non-catalogue item bucket), reproduced as config rules.
   - Pairwise P/R on A: suppliers 1.000 / 1.000, items 0.9998 / 0.9998. The generic C path
     is tuned on A's noisy tool names (P = R = 1.000) and with A's codes hidden (items P 0.97,
     R 0.99).
5. **Classification into an unseen tree (no labels).**
   - Each unique line text is expanded (SAP abbreviations lexicon) and embedded with
     multilingual `BAAI/bge-m3` (disk cache).
   - It is compared with every leaf as a full path and as a leaf name; the softmax is blended
     80/20 with a character TF-IDF view.
   - A supplier prior spreads confident labels to a supplier's vague lines; an item prior is
     used for blank/vague lines.
   - Top-down decoding: L1 by summed leaf probability, then L2 inside it, and so on.
   - The confidence is calibrated with isotonic curves fitted on A and B.
   - When the same company's labels exist (A, B, and therefore A2, B2), a kNN memory weighted
     by similarity is added. C runs without it.
   - Offline fallback: TF-IDF only, used automatically if the model can't load.

**Measured accuracy** (`analysis/eval_classify.py`, spend-weighted / line-weighted):

| | L1 | L2 | L3 | Calibration ECE |
|---|---|---|---|---|
| **A, labels hidden** (tree only) | **0.685** / 0.517 | **0.396** / 0.269 | **0.312** / 0.210 | 0.030 |
| B, labels hidden (stand-in for C) | 0.917 / 0.784 | 0.339 / 0.500 | 0.259 / 0.329 | 0.043 |
| A, half the labels as memory (A2) | 0.945 / 0.923 | 0.885 / 0.853 | 0.874 / 0.839 | 0.028 |
| B, half the seed as memory (B2) | 0.990 / 0.958 | 0.990 / 0.955 | 0.988 / 0.920 | 0.187 (under-confident) |
| TF-IDF fallback, labels hidden | A .68 / B .52 | A .39 / B .52 | A .27 / B .41 | |

Tried and rejected, with numbers in DECISIONS.md: e5-small/base and mpnet (weaker),
self-training on pseudo-labels (lowered accuracy), an item prior on clear text (hurt A).

Why zero-shot leaf accuracy is limited:
- A's labels depend on which machine a spare part is for, which its text doesn't say.
- B's largest spend is aluminium coil lines described only by dimension codes; L1 is right,
  L2 often isn't.

**Runtime.**
- Cold cache (every text embedded): A 65,549 rows in 1 m 50 s, B 41,028 rows in 3 m 44 s, on
  a laptop CPU.
- Warm cache: about 15 s.
- The 60-minute limit for 25,000 lines is far away.

**LLM cost.** None: the tool uses no API and needs no key, so the cost per 1,000 lines is 0.
An LLM re-ranker was deliberately left out because no key will be available at the blind test
and a local path was required anyway.

**Checks.**
- 132 pytest tests on synthetic data (VAT, FX direction and traps, dates, PO/invoice
  normalisation, matching scenarios, lexicon hits and misses, de-dup, tree loading, decoding,
  output contract, profile).
- Every run checks the output contract: rows = inputs, unique ids, spend 0 when excluded,
  category paths exactly as in the tree.

**New company.** `python -m spendpipe profile <file> --tree <tree> --out configs/company_C.yaml`
drafts the config (columns, date/number formats, FX table, status values, intercompany
preview). Rehearsed on B as an unknown company: identical spend on all 41,028 lines with no
edits.

**AI tools:** Claude Code was used as a pair-programmer for code, analysis queries and docs.
All numbers come from runs on the data.
