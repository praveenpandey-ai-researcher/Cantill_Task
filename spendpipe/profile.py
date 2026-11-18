"""`python -m spendpipe profile <file> [--tree <tree file>] [--out configs/company_C.yaml]`

Scans a new invoice-line file and drafts a config: guesses which column is the
id, amount, date, supplier, description, status, currency, FX rate, GL account,
PO, ... from column NAMES and CONTENT, detects date and number formats, shows
status values and how the built-in rules would treat them, and previews
intercompany / tax hits. Prints a short profile and writes a draft YAML with
comments to review. Nothing is guessed silently: every guess is printed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from spendpipe.clean import parse_amount, parse_date
from spendpipe.exclude import DEFAULT_STATUS
from spendpipe.ingest import read_table

# Name hints per canonical field (lower-case regex on the column name).
NAME_HINTS = {
    "record_id": r"^(line_?id|row_?id|id)$|line.?id|record.?id",
    "amount": r"amount|amt|value|net|spend|total|betrag|importe|valor|montant",
    "date": r"invoice.?date|inv.?date|doc.?date|document.?date|^date$|fecha|datum|data",
    "posting_date": r"post(ing)?.?date|gl.?date|booking",
    "supplier": r"supplier.?name|vendor.?name|supplier$|vendor$|creditor|proveedor|fornecedor|lieferant",
    "supplier_id": r"(supplier|vendor|creditor).?(id|no|number|code|num)|lifnr",
    "description": r"desc|text|item.?name|line.?text|narrative|memo|descripci|bezeichnung|libell",
    "item_code": r"material|item.?(code|no|number|id)|sku|part.?(no|number)|product.?code|matnr",
    "quantity": r"qty|quantity|menge|cantidad",
    "status": r"status|state|approval|estado",
    "account": r"(gl|ledger|account|acct|cuenta|konto).?(no|code|number|id)?$|^gl",
    "account_desc": r"(gl|account|acct).?(desc|name|text)",
    "cost_centre": r"cost.?cent|cc$|kostenstelle|centro",
    "po_number": r"po.?(no|number|num|ref)?$|purchase.?order|^po",
    "po_line": r"po.?(line|item)",
    "invoice_number": r"inv(oice)?.?(no|number|num|ref|id)",
    "currency": r"currency|ccy|curr|waers|moneda|devise",
    "fx_rate": r"(fx|exchange|currency).?rate|kurs",
    "vat_rate": r"vat|tax.?rate|gst.?rate|iva",
    "document_type": r"doc(ument)?.?type",
    "site": r"plant|site|location|werk",
}


_STATUS_WORDS = {w.lower() for ws in DEFAULT_STATUS.values() for w in ws}


def _shape(s: str) -> str:
    return re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "9", s))


def column_stats(col: pd.Series) -> dict:
    s = col.astype(str).str.strip()
    nonblank = s[s != ""]
    sample = nonblank.sample(min(len(nonblank), 2000), random_state=0) if len(nonblank) else nonblank
    num = sample.map(parse_amount)
    dates = sample.map(lambda v: parse_date(v) is not None)
    return {
        "blank_share": 1 - len(nonblank) / max(len(s), 1),
        "unique_share": nonblank.nunique() / max(len(nonblank), 1),
        "n_unique": int(nonblank.nunique()),
        "numeric_share": float(num.notna().mean()) if len(sample) else 0.0,
        "date_share": float(dates.mean()) if len(sample) else 0.0,
        "avg_len": float(nonblank.str.len().mean()) if len(nonblank) else 0.0,
        "alpha_share": float(sample.str.contains(r"[A-Za-z]{3}").mean()) if len(sample) else 0.0,
        "ccy_like": float(sample.str.fullmatch(r"[A-Z]{3}").mean()) if len(sample) else 0.0,
        "status_words": float(sample.str.lower().str.strip().isin(_STATUS_WORDS).mean()) if len(sample) else 0.0,
        "shapes": nonblank.map(_shape).value_counts().head(4).to_dict(),
    }


def guess_columns(df: pd.DataFrame) -> tuple[dict, dict]:
    """Score every (field, column) pair from name hints + content; pick the best
    column per field without reusing columns. Returns (mapping, stats)."""
    stats = {c: column_stats(df[c]) for c in df.columns}
    scores: dict[tuple[str, str], float] = {}
    for f, pat in NAME_HINTS.items():
        for c, st in stats.items():
            name = c.lower()
            sc = 3.0 if re.search(pat, name) else 0.0
            if st["blank_share"] > 0.98:
                sc -= 2
            if f == "record_id":
                sc += 2.0 * (st["unique_share"] > 0.999) - 1.0 * (st["blank_share"] > 0)
            elif f == "amount":
                sc += 2.0 * (st["numeric_share"] > 0.95) - 2.0 * (st["date_share"] > 0.5) \
                    + 0.5 * (st["unique_share"] > 0.2) - 2.0 * bool(re.search(r"rate|qty|quant|id$|_id", name))
            elif f in ("date", "posting_date"):
                sc += 2.0 * (st["date_share"] > 0.8) - 1.5 * (st["numeric_share"] > 0.9 and st["date_share"] < 0.8)
                sc -= 1.5 * bool(re.search(r"due|paid|pay|creat|deliver", name))
            elif f == "description":
                sc += 1.0 * (st["avg_len"] > 15) + 1.0 * (st["alpha_share"] > 0.5) + 0.5 * (st["unique_share"] > 0.05)
            elif f == "supplier":
                sc += 1.0 * (st["alpha_share"] > 0.7) - 1.0 * (st["numeric_share"] > 0.5)
            elif f == "currency":
                sc += 3.0 * (st["ccy_like"] > 0.9)
            elif f == "fx_rate":
                sc += 1.0 * (st["numeric_share"] > 0.9)
            elif f == "status":
                # must actually contain approval words (not e.g. 'On Contract')
                sc += 1.0 * (st["n_unique"] <= 15) + (2.0 if st["status_words"] > 0.3 else -3.0)
            elif f == "account_desc":
                sc += 0.5 * (st["alpha_share"] > 0.3)
            elif f == "vat_rate":
                sc += 0.5 * (st["numeric_share"] > 0.8) + 0.5 * (st["n_unique"] <= 20)
            scores[(f, c)] = sc
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for (f, c), sc in sorted(scores.items(), key=lambda kv: -kv[1]):
        if sc >= 3.0 and f not in mapping and c not in used:
            mapping[f] = c
            used.add(c)
    if "date" not in mapping and "posting_date" in mapping:
        mapping["date"] = mapping["posting_date"]          # only one date: use it for FX too
    if "record_id" not in mapping:      # content fallback: first unique, never-blank column
        for c, st in stats.items():
            if c not in used and st["blank_share"] == 0 and st["unique_share"] > 0.999:
                mapping["record_id"] = c
                break
    return mapping, stats


def date_formats(values: pd.Series) -> tuple[list[str], bool]:
    """Pick explicit formats from the value shapes; dayfirst from evidence."""
    s = values.astype(str).str.strip()
    s = s[s != ""]
    fmts, dayfirst = [], True
    shapes = s.map(_shape).value_counts()
    for shp in shapes.index[:6]:
        if re.fullmatch(r"9999-99-99.*", shp):
            fmts.append("%Y-%m-%d")
        elif re.fullmatch(r"99[/.\-]99[/.\-]9999", shp):
            sep = shp[2]
            parts = s[s.map(_shape) == shp].str.split(re.escape(sep), expand=True).astype(int)
            first_gt12, second_gt12 = int((parts[0] > 12).sum()), int((parts[1] > 12).sum())
            day_first = first_gt12 >= second_gt12
            dayfirst = day_first
            fmts.append(f"%d{sep}%m{sep}%Y" if day_first else f"%m{sep}%d{sep}%Y")
        elif re.fullmatch(r"99-aaa-99", shp):
            fmts.append("%d-%b-%y")
        elif re.fullmatch(r"99-aaa-9999", shp):
            fmts.append("%d-%b-%Y")
        elif re.fullmatch(r"99999999", shp):
            fmts.append("%Y%m%d")
    return list(dict.fromkeys(fmts)), dayfirst


def profile(path: str, tree: str | None = None, out: str | None = None, company: str = "C",
            reporting_currency: str = "USD") -> str:
    p = Path(path).resolve()
    df = read_table(p)
    mapping, stats = guess_columns(df)
    lines = [f"# profile of {p.name}: {len(df):,} rows, {df.shape[1]} columns", ""]
    lines.append(f"{'column':32s} {'blank':>6s} {'uniq':>7s} {'num':>5s} {'date':>5s}  shapes")
    for c, st in stats.items():
        lines.append(f"{c[:32]:32s} {st['blank_share']:6.1%} {st['n_unique']:7d} {st['numeric_share']:5.2f} "
                     f"{st['date_share']:5.2f}  {list(st['shapes'])[:3]}")
    lines += ["", "# guessed mapping (CHECK THESE):"]
    for f in NAME_HINTS:
        lines.append(f"  {f:15s} -> {mapping.get(f, '(not found)')}")

    # status values
    status_cfg = {}
    if "status" in mapping:
        vals = df[mapping["status"]].astype(str).str.strip().value_counts()
        known = {k: {v.lower() for v in vs} for k, vs in DEFAULT_STATUS.items()}
        lines += ["", "# status values and built-in treatment:"]
        for v, n in vals.items():
            norm = re.sub(r"[\s_\-]+", " ", v.lower()).strip()
            how = next((k for k, vs in known.items() if norm in vs), "UNKNOWN -> NOT_APPROVED")
            lines.append(f"  {v!r:30s} {n:7d}  {how}")
            if how.startswith("UNKNOWN"):
                status_cfg.setdefault("NOT_APPROVED", []).append(v)
    # date format
    fmts, dayfirst = ([], True)
    if "date" in mapping:
        fmts, dayfirst = date_formats(df[mapping["date"]])
        lines += ["", f"# date formats guessed: {fmts} dayfirst={dayfirst}"]
    # amounts: decimal comma?
    decimal = "auto"
    if "amount" in mapping:
        a = df[mapping["amount"]].astype(str)
        comma_dec = a.str.contains(r",\d{1,2}$").mean()
        dot_dec = a.str.contains(r"\.\d{1,2}$").mean()
        decimal = "," if comma_dec > dot_dec * 2 and comma_dec > 0.2 else "auto"
        lines.append(f"# amount: decimal comma share {comma_dec:.2f}, dot {dot_dec:.2f} -> decimal: {decimal}")
        neg = (a.map(parse_amount) < 0).mean()
        lines.append(f"# amount: {neg:.1%} negative (credit notes / reversals)")
    # currency
    if "currency" in mapping:
        lines.append("# currencies: " + str(df[mapping["currency"]].value_counts().head(12).to_dict()))
    # intercompany / tax preview
    ic_pat = re.compile(r"^(IC\d|IC\s|I/C\b|Intercompany|Inter-company)", re.I)
    for f in ("account_desc", "account", "supplier"):
        if f in mapping:
            v = df[mapping[f]].astype(str)
            hits = v[v.str.match(ic_pat)].value_counts()
            if len(hits):
                lines.append(f"# intercompany-looking {f} values: {hits.head(10).to_dict()}")
    if tree:
        t = read_table(Path(tree))
        lines.append(f"# tree {Path(tree).name}: {len(t)} rows, columns {list(t.columns)}")

    fx = find_fx_table(p)
    if fx:
        lines.append(f"# FX table found next to the data: {fx['file']} {fx}")
    elif "currency" in mapping:
        lines.append("# WARNING: currency column but no FX table found; lines without a rate get FX_MISSING")

    cfg_text = draft_config(p, mapping, fmts, dayfirst, decimal, status_cfg, tree, company,
                            reporting_currency, fx)
    report = "\n".join(lines)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(cfg_text, encoding="utf-8", newline="\n")
        report += f"\n\n# draft config written to {out} -- review every line before running"
    else:
        report += "\n\n" + cfg_text
    return report


def find_fx_table(p: Path) -> dict | None:
    """Look next to the data file for a monthly FX table (currency, month, rate)."""
    for f in sorted(p.parent.glob("*.csv")):
        if f == p or not re.search(r"fx|rate|exchange|currenc", f.name, re.I):
            continue
        t = read_table(f)
        cols = {c.lower(): c for c in t.columns}
        ccy = next((cols[c] for c in cols if re.search(r"curr|ccy", c)), None)
        month = next((cols[c] for c in cols if re.search(r"month|period|date", c)), None)
        rate = next((cols[c] for c in cols if re.search(r"rate|per_usd|units|value", c)), None)
        if ccy and month and rate:
            return {"file": f.name, "currency_column": ccy, "month_column": month, "rate_column": rate}
    return None


def draft_config(p: Path, m: dict, fmts, dayfirst, decimal, status_cfg, tree, company,
                 reporting_currency, fx: dict | None = None) -> str:
    def col(f, comment=""):
        v = m.get(f)
        line = f"      {f}: {v}" if v else f"      # {f}: <column>"
        return line + (f"   # {comment}" if comment else "")

    has_rate = "fx_rate" in m
    data_dir = Path("../..") / p.parent.name
    lines = [
        f"# Company {company}: DRAFT written by `python -m spendpipe profile`. Review every line.",
        f"company: {company}",
        f"reporting_currency: {reporting_currency}",
        f"data_dir: {data_dir.as_posix()}        # relative to this config; or pass --data-dir",
        "",
        "taxonomy:",
        f"  file: {Path(tree).name if tree else '<tree file>'}",
        "",
    ]
    if has_rate or "currency" in m or fx:
        lines += ["fx:"]
        if fx:
            lines += [f"  table_file: {fx['file']}",
                      f"  currency_column: {fx['currency_column']}",
                      f"  month_column: {fx['month_column']}",
                      f"  rate_column: {fx['rate_column']}"]
        else:
            lines += ["  table_file: null            # monthly FX table, if one is provided"]
        lines += ["  rate_meaning: units_per_reporting   # rate = units of line currency per 1 USD -> divide",
                  "  max_rate_deviation: 5", ""]
    lines += [
        "dates:",
        '  min: "2015-01-01"',
        '  max: "2026-12-31"',
        "",
        "sources:",
        "  - name: INV",
        f"    file: {p.name}",
        "    role: primary",
        '    id_prefix: ""',
        "    columns:",
    ]
    for f in NAME_HINTS:
        lines.append(col(f))
    lines += [f"    date_formats: {fmts}", f"    dayfirst: {str(dayfirst).lower()}"]
    if decimal != "auto":
        lines.append(f'    decimal: "{decimal}"')
    lines += ["    amount:", f"      gross: {str('vat_rate' in m).lower()}            # true only if amounts include VAT"]
    if "vat_rate" in m:
        lines.append("      default_vat_rate: null   # rate to use when the VAT cell is blank")
    if "currency" not in m:
        lines += ["    currency:", f"      fixed: {reporting_currency}"]
    if status_cfg:
        lines += ["    status:", "      # unknown values found; move each to include / CANCELLED / NOT_APPROVED"]
        for k, vs in status_cfg.items():
            lines.append(f"      {k}: {vs}")
    lines += [
        "",
        "matching:",
        "  enabled: false",
        "  exact_duplicates: true     # rows identical in every field but the id -> DUPLICATE",
        "",
        "exclusions:",
        "  non_procurement:",
        "    lexicon: lexicons/non_procurement.yaml",
        "    text_fields: [description]",
        "    account_desc_patterns: ['^IC\\d', '^IC\\s', '^I/C\\b', '^Intercompany\\b']",
        "    accounts: []             # GL codes that are not procurement (taxes, interco...)",
        "    supplier_patterns: []    # e.g. the company's own group name -> intercompany",
        "",
        "dedup:",
        "  supplier:",
        "    threshold: 92",
        "  item:",
        "    no_code: text",
        "",
        "classify:",
        "  model: BAAI/bge-m3",
        "  text_fields: [description]",
        "  # no labels for a new company: zero-shot + supplier/item priors",
        "",
    ]
    return "\n".join(lines)
