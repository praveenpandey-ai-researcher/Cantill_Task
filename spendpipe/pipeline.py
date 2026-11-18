"""Runs the stages in order. Each stage takes and returns the canonical frame.

    ingest -> clean -> status -> match (count once) -> non-procurement
    -> finalise exclusions -> de-dup keys -> classify -> output + report

The optional, model-heavy stages never stop a run: if classification fails
it is retried with the offline TF-IDF backend, then left blank; if de-dup
fails, simple normalised names / codes are used as keys. Problems are logged.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from spendpipe.classify import classify
from spendpipe.clean import clean, summarise
from spendpipe.config import Config
from spendpipe.dedup import dedup
from spendpipe.exclude import apply_non_procurement, apply_status, finalise
from spendpipe.ingest import ingest
from spendpipe.match import match
from spendpipe.normalise import norm_supplier

log = logging.getLogger(__name__)


def _safe_dedup(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    try:
        return dedup(df, cfg)
    except Exception:                                  # noqa: BLE001
        log.exception("de-dup failed; falling back to normalised names / codes")
        df = df.copy()
        df["supplier_key"] = df["supplier"].map(norm_supplier).where(df["supplier"] != "", df["supplier_id"])
        df["item_key"] = df["item_code"]
        return df


def _safe_classify(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    try:
        return classify(df, cfg)
    except Exception:                                  # noqa: BLE001
        log.exception("classification failed; retrying with the offline TF-IDF backend")
    try:
        cfg.raw.setdefault("classify", {})["backend"] = "tfidf"
        return classify(df, cfg)
    except Exception:                                  # noqa: BLE001
        log.exception("offline classification failed too; categories left blank")
        df = df.copy()
        for c in ("l1", "l2", "l3", "l4"):
            df[c] = ""
        df["confidence"] = 0.0
        return df, {"error": "classification failed"}


def run(cfg: Config, stop_before: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Run all stages. `stop_before='classify'` returns the frame just before
    classification (used by the evaluation scripts to try classifier settings)."""
    report: dict = {"company": cfg.company, "timings_s": {}}
    t0 = time.time()

    def timed(name, fn, *a):
        t = time.time()
        res = fn(*a)
        report["timings_s"][name] = round(time.time() - t, 2)
        return res

    df = timed("ingest", ingest, cfg)
    report["input_rows"] = int(len(df))
    df = timed("clean", clean, df, cfg)
    report["clean"] = summarise(df, cfg.reporting_currency)

    df = timed("status", apply_status, df, cfg)
    report["status"] = {f"{s}:{r or 'counts'}": int(n) for (s, r), n in
                        df.groupby(["source", "status_reason"]).size().items()}

    df, report["matching"] = timed("match", match, df, cfg)

    df = timed("non_procurement", apply_non_procurement, df, cfg)
    df = finalise(df)
    report["exclusions"] = {
        "by_reason": {k or "included": int(v) for k, v in df.exclusion_reason.value_counts().items()},
        "non_procurement_rules": {k: int(v) for k, v in df.np_rule[df.np_rule != ""].value_counts().items()},
    }

    df = timed("dedup", _safe_dedup, df, cfg)
    report["dedup"] = {"supplier_keys": int(df.supplier_key[df.supplier_key != ""].nunique()),
                       "item_keys": int(df.item_key[df.item_key != ""].nunique()),
                       "lines_without_item_key": int((df.item_key == "").sum())}
    if stop_before == "classify":
        return df, report

    df, report["classify"] = timed("classify", _safe_classify, df, cfg)
    report["timings_s"]["total"] = round(time.time() - t0, 2)
    return df, report


def run_to_file(cfg: Config, out_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Run the pipeline, write the output CSV and its data-quality report."""
    from spendpipe.output import (build_output, check_output, check_paths, quality_report,
                                  summary_text, write, write_report)
    from spendpipe.taxonomy import load_tree
    df, report = run(cfg)
    out = build_output(df)
    check_output(out, report["input_rows"])
    report["category_paths_not_in_tree"] = check_paths(out, load_tree(cfg))
    path = write(out, out_path)
    rep = quality_report(df, out, report)
    rpath = write_report(rep, out_path)
    log.info("wrote %s (%d rows) and %s\n%s", path, len(out), rpath.name,
             summary_text(rep, cfg.reporting_currency))
    return out, rep
