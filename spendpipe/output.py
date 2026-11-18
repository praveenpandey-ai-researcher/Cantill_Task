"""Write the final output CSV and the data-quality report.

Output columns (exact order, one row per input record, excluded rows too):
    record_id, include, exclusion_reason, supplier_key, item_key, spend,
    l1, l2, l3, l4, confidence

spend      net spend in the reporting currency; 0 when excluded or unparsable
l1..l4     tree names exactly as in the taxonomy file; blank below the branch's
           lowest level and for excluded lines
confidence included lines: calibrated classification confidence;
           excluded lines: how sure the exclusion is (by the rule that fired)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COLUMNS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
           "l1", "l2", "l3", "l4", "confidence"]

# How certain each exclusion rule is (generic, not company data).
EXCLUSION_CONFIDENCE = {
    "status": 1.0,                 # the system says cancelled / rejected / on hold
    "exact_row": 0.99,             # every field identical
    "po_line_amount": 0.99,
    "po_amount": 0.97,
    "reexported_invoice": 0.9,
    "supplier_amount": 0.85,
    "po_text": 0.85,
    "po_text_any": 0.75,
    "account": 0.99,               # GL account rule
    "account_desc": 0.97,
    "supplier_id": 0.95,
    "supplier_pattern": 0.9,
    "text": 0.9,                   # lexicon phrase in the description
}


def exclusion_confidence(df: pd.DataFrame) -> pd.Series:
    conf = pd.Series(np.nan, index=df.index)
    status = df.exclusion_reason.isin(["CANCELLED", "NOT_APPROVED"])
    conf[status] = EXCLUSION_CONFIDENCE["status"]
    dup = df.exclusion_reason == "DUPLICATE"
    conf[dup] = df.loc[dup, "match_pass"].map(EXCLUSION_CONFIDENCE).fillna(0.8)
    npr = df.exclusion_reason == "NON_PROCUREMENT"
    rule = df.loc[npr, "np_rule"].str.split(":").str[0]
    conf[npr] = rule.map(EXCLUSION_CONFIDENCE).fillna(0.8)
    return conf


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["record_id"] = df["record_id"]
    out["include"] = df["include"].astype(int)
    out["exclusion_reason"] = df["exclusion_reason"].fillna("")
    out["supplier_key"] = df["supplier_key"].fillna("")
    out["item_key"] = df["item_key"].fillna("")
    spend = pd.to_numeric(df["spend_raw"], errors="coerce").fillna(0.0)
    out["spend"] = np.where(out["include"] == 1, spend, 0.0).round(2)
    for c in ("l1", "l2", "l3", "l4"):
        out[c] = df[c].fillna("") if c in df else ""
        out.loc[out["include"] == 0, c] = ""
    conf = pd.to_numeric(df.get("confidence", 0.0), errors="coerce").fillna(0.0)
    excl = exclusion_confidence(df)
    out["confidence"] = np.where(out["include"] == 1, conf, excl.fillna(0.8)).clip(0, 1).round(4)
    return out[COLUMNS]


def check_output(out: pd.DataFrame, n_input: int) -> None:
    """Hard checks: the file must have exactly one row per input record."""
    problems = []
    if len(out) != n_input:
        problems.append(f"{len(out)} output rows for {n_input} input records")
    if out.record_id.duplicated().any():
        problems.append(f"{int(out.record_id.duplicated().sum())} duplicated record_id")
    if (out.record_id == "").any():
        problems.append("blank record_id")
    if ((out.include == 1) & (out.exclusion_reason != "")).any():
        problems.append("included lines with an exclusion reason")
    if ((out.include == 0) & (out.spend != 0)).any():
        problems.append("excluded lines with spend")
    if problems:
        raise RuntimeError("output check failed: " + "; ".join(problems))


def check_paths(out: pd.DataFrame, tree) -> int:
    """Every non-blank l1..l4 path must be a leaf path of the tree, written
    exactly as in the tree file. Returns the number of bad rows (logged)."""
    leaves = {lf.path for lf in tree.leaves}
    paths = out[["l1", "l2", "l3", "l4"]].apply(lambda r: tuple(v for v in r if v), axis=1)
    bad = paths.map(lambda p: bool(p) and p not in leaves)
    if bad.any():
        log.error("%d rows have a category path that is not a leaf of the tree", int(bad.sum()))
    return int(bad.sum())


def write(out: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = out.copy()
    out["spend"] = out["spend"].map(lambda x: f"{x:.2f}")
    out["confidence"] = out["confidence"].map(lambda x: f"{x:.4f}")
    out.to_csv(path, index=False, encoding="utf-8")
    return path


def quality_report(df: pd.DataFrame, out: pd.DataFrame, report: dict) -> dict:
    """Aggregates only: safe to commit next to the output."""
    flags = df["dq_flags"].fillna("").str.split(";").explode()
    flags = flags[flags != ""].value_counts()
    inc = out.include == 1
    rep = {
        "company": report.get("company"),
        "rows": int(len(out)),
        "included": int(inc.sum()),
        "excluded_by_reason": {k: int(v) for k, v in out.exclusion_reason[~inc].value_counts().items()},
        "spend_total_included": round(float(out.spend.sum()), 2),
        "spend_by_l1": {k: round(float(v), 2) for k, v in
                        out[inc].groupby("l1").spend.sum().sort_values(ascending=False).items()},
        "data_quality_flags": {k: int(v) for k, v in flags.items()},
        "supplier_keys": int(out.supplier_key[inc & (out.supplier_key != "")].nunique()),
        "item_keys": int(out.item_key[inc & (out.item_key != "")].nunique()),
        "classified_lines": int((inc & (out.l1 != "")).sum()),
        "mean_confidence_included": round(float(out.confidence[inc].mean()), 3) if inc.any() else None,
        "low_confidence_included_lt_0.3": int((inc & (out.confidence < 0.3)).sum()),
    }
    for k in ("category_paths_not_in_tree", "clean", "status", "matching", "exclusions", "dedup",
              "classify", "timings_s"):
        if k in report:
            rep[k] = report[k]
    return rep


def write_report(rep: dict, out_path: str | Path) -> Path:
    p = Path(out_path)
    rp = p.with_name(p.stem + "_report.json")
    rp.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    return rp


def summary_text(rep: dict, currency: str) -> str:
    lines = [
        f"company {rep['company']}: {rep['rows']:,} rows, {rep['included']:,} included",
        "excluded: " + ", ".join(f"{k} {v:,}" for k, v in rep["excluded_by_reason"].items()),
        f"included spend: {rep['spend_total_included']:,.2f} {currency}",
        f"supplier keys {rep['supplier_keys']:,}, item keys {rep['item_keys']:,}, "
        f"mean confidence {rep['mean_confidence_included']}",
        "data-quality flags: " + (", ".join(f"{k} {v:,}" for k, v in rep["data_quality_flags"].items()) or "none"),
        f"runtime: {rep.get('timings_s', {}).get('total')} s",
    ]
    return "\n".join(lines)
