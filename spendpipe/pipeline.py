"""Runs the stages in order. Each stage takes and returns the canonical frame."""
from __future__ import annotations

import json
import logging
import time

import pandas as pd

from spendpipe.clean import clean, summarise
from spendpipe.config import Config
from spendpipe.ingest import ingest

log = logging.getLogger(__name__)


def run(cfg: Config) -> tuple[pd.DataFrame, dict]:
    report: dict = {"company": cfg.company, "timings_s": {}}
    t0 = time.time()

    def timed(name, fn, *a):
        t = time.time()
        res = fn(*a)
        report["timings_s"][name] = round(time.time() - t, 2)
        return res

    df = timed("ingest", ingest, cfg)
    df = timed("clean", clean, df, cfg)
    report["clean"] = summarise(df, cfg.reporting_currency)

    report["timings_s"]["total"] = round(time.time() - t0, 2)
    log.info("Data quality:\n%s", json.dumps(report, indent=2, default=str))
    return df, report
