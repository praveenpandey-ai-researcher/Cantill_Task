# Blind-test runbook (Thu 19 Nov, 10:00 to 12:00 IST)

The rule: **no code changes after 10:00.** Only a new config for C and the three output files
go into `submission/`. Every command below uses the frozen code.

## 0. Before 10:00
- `git status` is clean, and the last commit is pushed.
- `python -m pytest -q` passes.
- The model is cached: run once on A or B and check the log says `embedding cache: N vectors`.

## 1. Unpack the data (5 min)
Put each data set in its own folder next to the kit, e.g. `../A2`, `../B2`, `../C`. Do not put
data inside the repo.

## 2. A2 and B2: same layouts, existing configs (10 min)
```bash
python -m spendpipe run --config configs/company_A.yaml --data-dir ../A2 --out submission/praveenpandey__A2.csv
python -m spendpipe run --config configs/company_B.yaml --data-dir ../B2 --out submission/praveenpandey__B2.csv
```
- Reference files missing from the A2/B2 folder (`taxonomy_*.csv`, `fx_rates_usd.csv`) are
  taken from the original `company_A` / `company_B` folder automatically, with a warning. If a
  *new* FX table is supplied, keep it in the B2 folder so it is used.
- Data files are never taken from the old folder: if the A2 files have different names, the
  run stops with "file not found". Rename the files to `erp_goods_receipts.csv` /
  `invoice_tool_export.csv` / `invoice_lines.csv`, which is a data step, not a code change.
- The labelled memory (A's labels, B's seed) is read from the original `../../company_A` and
  `../../company_B` folders automatically.
- Check the console summary: row count = input rows, the exclusion counts look like A/B's,
  `category paths not in tree` = 0 in the report.

## 3. C: new company (60-80 min)
1. Draft the config:
   ```bash
   python -m spendpipe profile ../C/<invoice file>.csv --tree ../C/<tree file>.csv --out configs/company_C.yaml
   ```
2. Review `configs/company_C.yaml` against the printed profile:
   - **columns**: id, amount, date, supplier / supplier id, description, currency, FX rate,
     GL account / description, PO, material code. Fix wrong guesses and add a missing
     mapping. A list of columns joins texts (`description: [col_a, col_b]`).
   - **data_dir** points at `../C` (or pass `--data-dir ../C`).
   - **amount.gross**: true only if amounts include VAT (then set `vat_rate` /
     `default_vat_rate`).
   - **currency**: reporting currency is USD. Check `fx.table_file` if a rate table exists.
     Use `currency: {fixed: USD}` if there is no currency column.
   - **status**: move each listed unknown value to `include`, `CANCELLED` or `NOT_APPROVED`.
   - **dates**: `date_formats` and `dayfirst` as printed. Widen `dates.min/max` if the data is
     older or newer.
   - **exclusions**: add GL accounts / descriptions that are taxes, intercompany, finance
     costs etc. (`accounts`, `account_desc_patterns`), and the company's own group name in
     `supplier_patterns` (intercompany).
   - **classify.text_fields**: the description column(s). Add `context_fields` only if useful.
   - **classify.reference**: leave unset (C has no labels).
3. Run:
   ```bash
   python -m spendpipe run --config configs/company_C.yaml --out submission/praveenpandey__C.csv
   ```
4. Read `submission/praveenpandey__C_report.json`:
   - excluded counts look plausible
   - `FX_MISSING` is near 0
   - `DATE_UNPARSABLE` is near 0
   - spend by L1 makes sense
   - `category_paths_not_in_tree` = 0

   Fix the config, never the code, and rerun.

## 4. Submit (by 12:00)
```bash
git add configs/company_C.yaml submission/
git commit -m "Blind test: C config and A2/B2/C outputs"
git push
```
Only the C config and the `submission/` files change. Check with `git diff --stat HEAD~1` that
no `.py` file is in the commit.
