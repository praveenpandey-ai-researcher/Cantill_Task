"""Evidence for the count-once matching on company A (no ground truth exists).

There are no labels for which tool lines duplicate which ERP lines, so the
checks are proxies, printed as aggregates only:
  1. Per pass: |date gap| (when both dates exist), same quantity, text
     similarity, and whether the tool supplier maps to the same golden
     supplier key as the matched ERP line (via labels_A on ERP names).
  2. Tool-only lines: split by PO situation.
  3. Spend reconciliation.

    python analysis/eval_matching.py --config configs/company_A.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from spendpipe.config import load_config
from spendpipe.normalise import norm_po, norm_supplier, norm_text, squash
from spendpipe.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/company_A.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(args.config)
    df, rep = run(cfg)
    m = cfg.section("matching")
    prim, sec = m.get("primary", "ERP"), m.get("secondary", "TOOL")
    P = df[df.source == prim].set_index("record_id")
    S = df[df.source == sec].copy()

    # golden supplier key per normalised ERP supplier name (from labels)
    lab = pd.read_csv(cfg.file(cfg.section("labels")["file"]), dtype=str, keep_default_na=False)
    P = P.join(lab.set_index("record_id")[["supplier_key"]])
    name2key = (P.assign(n=P.supplier.map(lambda s: squash(norm_supplier(s))))
                .groupby("n").supplier_key.agg(lambda s: s.mode().iat[0]))

    print("pass               lines  both_dates  gap_med  gap_p90  same_qty  text>=90  same_golden_supplier")
    for pas, g in S[S.match_id != ""].groupby("match_pass"):
        if pas in ("reexported_invoice", "exact_row"):
            continue
        e = P.loc[g.match_id]
        gap = pd.Series((g.date_parsed.values - e.date_parsed.values)).dt.days.abs()
        qty = (pd.to_numeric(g.quantity, errors="coerce").values
               == pd.to_numeric(e.quantity, errors="coerce").values)
        txt = np.array([fuzz.token_set_ratio(norm_text(a), norm_text(b))
                        for a, b in zip(g.description, e.description)])
        tkey = g.supplier.map(lambda s: name2key.get(squash(norm_supplier(s)), None)).values
        known = pd.notna(tkey)
        same = (tkey[known] == e.supplier_key.values[known]).mean() if known.any() else np.nan
        print(f"{pas:17s} {len(g):6d} {gap.notna().sum():11d} {gap.median():8.0f} "
              f"{gap.quantile(.9):8.0f} {qty.mean():9.2f} {(txt >= 90).mean():9.2f} {same:21.3f}")

    pat = m.get("po_pattern")
    S["po"] = S.po_number.map(lambda v: norm_po(v, pat))
    erp_pos = set(P.po_number.map(lambda v: norm_po(v, pat)))
    only = S[(S.status_reason == "") & (S.dup_reason == "")]
    print("\ntool-only lines", len(only), "| no PO", int((only.po == "").sum()),
          "| PO in ERP", int(only.po.isin(erp_pos).sum()),
          "| PO not in ERP", int(((only.po != "") & ~only.po.isin(erp_pos)).sum()))
    print("tool exclusions:", S.status_reason.where(S.status_reason != "", S.dup_reason)
          .replace("", "included").value_counts().to_dict())
    erp_total, only_total = P.spend_raw.sum(), only.spend_raw.sum()
    print(f"spend: {prim} {erp_total:,.0f} + {sec}-only {only_total:,.0f} = {erp_total + only_total:,.0f}")
    print("matching stats:", rep["matching"])


if __name__ == "__main__":
    main()
