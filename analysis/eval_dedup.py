"""Supplier / item de-dup quality against A's golden keys (pairwise P/R),
plus threshold sweeps for the generic path used on a new company (C).

Pairwise metric: over all pairs of lines, precision = share of pairs we put
together that the client also puts together; recall = share of the client's
pairs that we put together.

    python analysis/eval_dedup.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from spendpipe.config import load_config
from spendpipe.dedup import components, fuzzy_name_edges, text_clusters
from spendpipe.normalise import norm_supplier, norm_text, squash
from spendpipe.pipeline import run


def n_pairs(s: pd.Series) -> int:
    c = s.value_counts()
    return int((c * (c - 1) // 2).sum())


def pairwise(pred: pd.Series, gold: pd.Series) -> tuple[float, float, float]:
    pred, gold = pred.astype(str).reset_index(drop=True), gold.astype(str).reset_index(drop=True)
    tp = n_pairs(pred + "\x1f" + gold)
    pp, gp = n_pairs(pred), n_pairs(gold)
    p = tp / pp if pp else 1.0
    r = tp / gp if gp else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config("configs/company_A.yaml")
    df, rep = run(cfg)
    lab = pd.read_csv(cfg.file(cfg.section("labels")["file"]), dtype=str, keep_default_na=False)
    m = df.merge(lab, on="record_id", suffixes=("", "_gold"))
    print(f"labelled ERP lines: {len(m)}  (dedup took {rep['timings_s']['dedup']} s)")

    print("\n== A with its config (client bucket rules on) ==")
    print("supplier_key  P/R/F", pairwise(m.supplier_key, m.supplier_key_gold))
    print("item_key      P/R/F (blank = one group)", pairwise(m.item_key, m.item_key_gold))
    k = m.item_key_gold != ""
    print("item_key      P/R/F (keyed lines only) ", pairwise(m.item_key[k], m.item_key_gold[k]))

    tool = df[df.source != "ERP"]
    print("tool lines: supplier keys shared with ERP lines:",
          f"{tool.supplier_key.isin(set(m.supplier_key)).mean():.3f}")

    # ---- generic path (what C gets): names only, no buckets --------------------
    big_s = m.supplier_key_gold.value_counts().index[0]
    real = m[m.supplier_key_gold != big_s]            # golden keys that are real suppliers
    names = real.supplier.map(lambda s: squash(norm_supplier(s)))
    uniq = sorted(set(names))
    print(f"\n== generic supplier name clustering ({len(uniq)} names, {len(real)} lines, "
          "catch-all removed) ==")
    for th in (80, 85, 88, 90, 92, 95, 100):
        comp = components(["N:" + n for n in uniq],
                          [("N:" + a, "N:" + b) for a, b in fuzzy_name_edges(uniq, th, 3)])
        pred = names.map(lambda n: comp["N:" + n])
        print(f"  ratio>={th:3d}  P/R/F", pairwise(pred, real.supplier_key_gold))

    big_i = m.item_key_gold[k].value_counts().index[0]
    items = m[k & (m.item_key_gold != big_i)]
    texts = items.description.map(norm_text)
    print(f"\n== generic item text clustering, codes hidden ({texts.nunique()} texts, "
          f"{len(items)} lines, catch-all removed) ==")
    print("  exact normalised text   P/R/F", pairwise(texts, items.item_key_gold))
    for th in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97):
        cl = text_clusters(texts.tolist(), th)
        print(f"  cosine>={th:.2f}           P/R/F", pairwise(texts.map(cl), items.item_key_gold))


if __name__ == "__main__":
    main()
