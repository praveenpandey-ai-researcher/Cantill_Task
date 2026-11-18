"""Score exclusions against labelled include/exclusion_reason (B's seed labels),
and list which distinct texts each lexicon rule caught (for false-positive review).

    python analysis/eval_exclusions.py --config configs/company_B.yaml
    python analysis/eval_exclusions.py --config configs/company_A.yaml   # no labels: hits only

Texts printed are generic descriptions (e.g. 'SALES TAX EXPENSE'), aggregated with counts.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from spendpipe.config import load_config
from spendpipe.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/company_B.yaml")
    ap.add_argument("--show", type=int, default=40, help="distinct texts to list per rule")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(args.config)
    df, rep = run(cfg)

    print("exclusions:", rep["exclusions"]["by_reason"])
    print("non-procurement rules:", rep["exclusions"]["non_procurement_rules"])

    text_hits = df[df.np_rule.str.startswith("text:")]
    for rule, g in text_hits.groupby("np_rule"):
        print(f"\n--- {rule}: {len(g)} lines")
        print(g.description.str.upper().value_counts().head(args.show).to_string())

    lab_cfg = cfg.section("labels") or {}
    if not lab_cfg.get("file") or "exclusion_reason" not in pd.read_csv(
            cfg.file(lab_cfg["file"]), nrows=0).columns:
        return
    lab = pd.read_csv(cfg.file(lab_cfg["file"]), dtype=str, keep_default_na=False)
    m = df.merge(lab[[lab_cfg.get("id_column", "line_id"), "exclusion_reason"]]
                 .rename(columns={lab_cfg.get("id_column", "line_id"): "record_id",
                                  "exclusion_reason": "label"}), on="record_id")
    print(f"\n=== against {len(m)} labelled lines ===")
    print(pd.crosstab(m.label.replace("", "included"), m.exclusion_reason.replace("", "included"),
                      rownames=["label"], colnames=["predicted"]))
    y = m.label == "NON_PROCUREMENT"
    p = m.np_reason == "NON_PROCUREMENT"
    tp = (y & p).sum()
    print(f"NON_PROCUREMENT precision {tp / max(p.sum(), 1):.3f}  recall {tp / max(y.sum(), 1):.3f}")
    fn = m[y & ~p]
    fp = m[~y & p]
    if len(fn):
        print("\nmissed (label NON_PROCUREMENT):")
        print(fn.groupby(["account_desc", fn.description.str.upper()]).size().head(args.show).to_string())
    if len(fp):
        print("\nextra (label included):")
        print(fp.groupby(["np_rule", fp.description.str.upper()]).size().head(args.show).to_string())


if __name__ == "__main__":
    main()
