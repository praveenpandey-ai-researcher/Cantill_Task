"""Stage 1 data forensics: reproduces the numbers quoted in DECISIONS.md.

Prints AGGREGATES ONLY (counts, shares, format shapes). It never prints raw rows,
so its output is safe to paste into docs.

    python analysis/explore_data.py --a ../company_A --b ../company_B
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def shape(s: str) -> str:
    """'INV/2023/123' -> 'aaa/9999/999': shows the format without the data."""
    return re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "9", s))


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def tool_vat(rate: pd.Series) -> pd.Series:
    """'15', '15%', '0.15' -> 0.15; '0' -> 0.0; blank -> 0.15 (brief default)."""
    r = pd.to_numeric(rate.str.replace("%", "", regex=False), errors="coerce")
    r = r.where(r <= 1, r / 100)
    return r.fillna(0.15)


def explore_a(d: Path) -> None:
    erp = read(d / "erp_goods_receipts.csv")
    tool = read(d / "invoice_tool_export.csv")
    lab = read(d / "labels_A.csv")

    section("A / ERP")
    print("rows", len(erp))
    print("blank posting_date", (erp.posting_date == "").sum(),
          "| blank vendor_id", (erp.vendor_id == "").sum(),
          "| same rows", ((erp.posting_date == "") == (erp.vendor_id == "")).all())
    v = erp.value_zar.astype(float)
    print("value_zar negative", (v < 0).sum(), "| zero", (v == 0).sum())
    print("document_currency", erp.document_currency.value_counts().to_dict())
    print("po_number shapes", erp.po_number.map(shape).value_counts().to_dict())
    print("blank material_number", (erp.material_number == "").sum())

    m = erp.assign(record_id="ERP:" + erp.line_id).merge(lab, on="record_id")
    print("supplier names", m.supplier_name.nunique(), "| golden supplier keys", m.supplier_key.nunique())
    big = m.supplier_key.value_counts().index[0]
    cat = m.supplier_key == big
    print(f"largest key: {cat.sum()} lines, {m[cat].supplier_name.nunique()} names, "
          f"blank vendor_id share {(m[cat].vendor_id == '').mean():.3f}")
    print("item_key blank", (m.item_key == "").sum(),
          "| == blank material_number", ((m.item_key == "") == (m.material_number == "")).all())
    print("label L3 == L2 share", round((lab.category_l3 == lab.category_l2).mean(), 3))

    section("A / TOOL")
    print("rows", len(tool))
    print("status", tool.status.value_counts().to_dict())
    print("vat_rate", tool.vat_rate.value_counts().to_dict())
    print("invoice_date shapes", tool.invoice_date.map(shape).value_counts().to_dict())
    sl = tool.invoice_date[tool.invoice_date.str.contains("/")].str.split("/", expand=True).astype(int)
    print("slash dates: part1>12", (sl[0] > 12).sum(), "| part2>12", (sl[1] > 12).sum())
    print("gross_amount shapes (top)", tool.gross_amount.map(shape).value_counts().head(6).to_dict())
    print("po_reference shapes", tool.po_reference.map(shape).value_counts().to_dict())
    print("invoice_number prefixes", tool.invoice_number.str.extract(r"^([^\d]*)")[0].value_counts().to_dict())

    # Cross-system overlap
    tool["po"] = tool.po_reference.str.extract(r"(\d{10})")[0]
    tool["net"] = (tool.gross_amount.str.replace(",", "").astype(float) / (1 + tool_vat(tool.vat_rate))).round(2)
    erp["v"] = v.round(2)
    pos = set(erp.po_number)
    by_po = erp.groupby("po_number").v.apply(lambda s: s.values)
    def amount_hit(po, amt):
        return po in by_po.index and np.any(np.abs(by_po[po] - amt) <= 0.02)
    has_po = tool.po.notna()
    in_erp = tool.po.isin(pos)
    hit = np.array([amount_hit(p, a) for p, a in zip(tool.po, tool.net)])
    print("tool lines: no PO", (~has_po).sum(), "| PO not in ERP", (has_po & ~in_erp).sum(),
          "| PO in ERP", in_erp.sum(), "| PO in ERP and net amount exact", (in_erp & hit).sum())

    # VAT semantics check on PO-matched lines
    g = tool.gross_amount.str.replace(",", "").astype(float)
    for label, mask in [("0", tool.vat_rate == "0"), ("blank", tool.vat_rate == "")]:
        sub = in_erp & mask
        as_is = np.mean([amount_hit(p, round(x, 2)) for p, x in zip(tool.po[sub], g[sub])])
        div = np.mean([amount_hit(p, round(x / 1.15, 2)) for p, x in zip(tool.po[sub], g[sub])])
        print(f"vat_rate {label}: match gross as-is {as_is:.3f} | match gross/1.15 {div:.3f}")

    core = tool.invoice_number.str.extract(r"(\d+)\s*$")[0].str.lstrip("0")
    key = [core, tool.supplier_name_as_captured, tool.line_text, tool.net]
    dup = pd.DataFrame(dict(zip(["c", "s", "t", "n"], key))).duplicated(keep="first")
    print("tool lines that repeat an earlier line (invoice core+supplier+text+net)", dup.sum())


def explore_b(d: Path) -> None:
    b = read(d / "invoice_lines.csv")
    seed = read(d / "seed_labels.csv")
    fx = read(d / "fx_rates_usd.csv")

    section("B / invoice lines")
    print("rows", len(b), "| columns", len(b.columns), "| status column present", "status" in b.columns)
    print("currencies", b.currency.nunique(), "| missing from FX table",
          sorted(set(b.currency) - set(fx.currency) - {"USD"}))
    rate = pd.to_numeric(b.currency_rate, errors="coerce")
    print("USD lines carrying a rate", ((b.currency == "USD") & rate.notna()).sum())
    print("non-USD lines without rate", ((b.currency != "USD") & rate.isna()).sum())
    tab = fx.assign(u=fx.units_per_usd.astype(float)).set_index(["currency", "month"]).u
    x = b[rate.notna() & (b.currency != "USD")]
    ratio = rate[x.index] / [tab.get((c, m[:7]), np.nan) for c, m in zip(x.currency, x.invoice_date)]
    print("given rate / table rate (invoice month) quantiles", ratio.quantile([.01, .5, .99]).round(3).to_dict())
    for c in ["posting_date", "invoice_date", "due_date"]:
        yrs = pd.to_datetime(b[c], errors="coerce").dt.year.value_counts().sort_index().to_dict()
        print(c, "years", yrs)
    a = pd.to_numeric(b.line_amount, errors="coerce")
    print("amount blank", a.isna().sum(), "| negative", (a < 0).sum(), "| zero", (a == 0).sum())
    print("description empty/space", (b.line_description.str.strip() == "").sum(),
          "| numbers only", b.line_description.str.strip().str.fullmatch(r"[\d\-\s]+").sum())
    cols = [c for c in b.columns if c != "line_id"]
    later = b.duplicated(cols, keep="first")
    print("exact duplicate rows (all fields but line_id)", later.sum(),
          "by system", b[later].source_system.value_counts().to_dict())
    ic = b.gl_account_description.str.match(r"^(IC\d|IC |I/C|Intercompany)")
    print("intercompany GL description lines", ic.sum())

    section("B / seed labels")
    m = b.merge(seed, on="line_id")
    print("exclusion_reason", seed.exclusion_reason.value_counts().to_dict())
    mic = m.gl_account_description.str.match(r"^(IC\d|IC |I/C|Intercompany)")
    print("NON_PROCUREMENT that are intercompany GL", (mic & (m.exclusion_reason == "NON_PROCUREMENT")).sum(),
          "| intercompany GL lines included", (mic & (m.include == "1")).sum())
    rest = m[~mic & (m.exclusion_reason == "NON_PROCUREMENT")]
    print("other NON_PROCUREMENT descriptions", rest.line_description.str.strip().value_counts().to_dict())
    m_dup = m.line_id.isin(b.line_id[b.duplicated(cols, keep=False)])
    print("seed rows inside exact-duplicate groups", m_dup.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="../company_A")
    ap.add_argument("--b", default="../company_B")
    args = ap.parse_args()
    explore_a(Path(args.a))
    explore_b(Path(args.b))


if __name__ == "__main__":
    main()
