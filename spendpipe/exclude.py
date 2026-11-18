"""Exclusion rules and reason codes.

Every line ends with `include` (1/0) and `exclusion_reason`:
    CANCELLED        cancelled invoice or line
    NOT_APPROVED     rejected, on hold, submitted, disputed...
    NON_PROCUREMENT  taxes, intercompany, finance... (stage 4)
    DUPLICATE        already counted in another record (stage 3, match.py)

When several apply, the first in this order wins (the record's own state comes
first; a cancelled invoice is reported as CANCELLED even if it also matches ERP):
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from spendpipe.config import Config

log = logging.getLogger(__name__)

PRECEDENCE = ["CANCELLED", "NOT_APPROVED", "DUPLICATE", "NON_PROCUREMENT"]

# Generic status words (several languages). A source config can extend them.
DEFAULT_STATUS = {
    "include": ["paid", "approved", "posted", "cleared", "complete", "completed", "closed",
                "pagado", "aprobado", "pago", "aprovado", "payé", "paye", "approuvé",
                "bezahlt", "genehmigt", "gebucht"],
    "CANCELLED": ["cancelled", "canceled", "cancel", "void", "voided", "reversed", "deleted",
                  "cancelado", "anulado", "annulé", "annule", "storniert", "storno"],
    "NOT_APPROVED": ["rejected", "on hold", "hold", "submitted", "disputed", "pending",
                     "draft", "in review", "parked", "blocked", "awaiting approval",
                     "rechazado", "pendiente", "rejeitado", "pendente", "rejeté", "en attente",
                     "abgelehnt", "gesperrt"],
}


def _norm_status(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s or "").strip().lower())


def status_reason(value: str, rules: dict, blank_means: str = "include",
                  unknown_means: str = "NOT_APPROVED") -> str:
    """'' if the status counts as spend, else CANCELLED / NOT_APPROVED.
    Brief: only approved or paid invoices count, so an unknown status is
    NOT_APPROVED by default. Blank status (system has none) counts by default."""
    s = _norm_status(value)
    if not s:
        return "" if blank_means == "include" else blank_means
    for reason in ("CANCELLED", "NOT_APPROVED", "include"):
        if s in rules.get(reason, []):
            return "" if reason == "include" else reason
    return "" if unknown_means == "include" else unknown_means


def apply_status(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Adds `status_reason` per line from each source's status rules."""
    df = df.copy()
    df["status_reason"] = ""
    for src in cfg.sources:
        scfg = src.get("status") or {}
        rules = {k: [_norm_status(x) for x in DEFAULT_STATUS[k] + list(scfg.get(k, []))]
                 for k in DEFAULT_STATUS}
        mask = df.source == src["name"]
        vals = df.loc[mask, "status"]
        mapping = {v: status_reason(v, rules, scfg.get("blank_means", "include"),
                                    scfg.get("unknown_means", "NOT_APPROVED"))
                   for v in vals.unique()}
        df.loc[mask, "status_reason"] = vals.map(mapping)
        unknown = [v for v in mapping if v and _norm_status(v) not in sum(rules.values(), [])]
        if unknown:
            log.warning("[%s] unknown status values treated as %s: %s", src["name"],
                        scfg.get("unknown_means", "NOT_APPROVED"), unknown[:10])
    return df


def combine_reasons(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Pick one exclusion reason per line from several candidate columns,
    following PRECEDENCE."""
    out = pd.Series("", index=df.index)
    for reason in reversed(PRECEDENCE):       # lowest priority first, overwritten later
        for c in columns:
            if c in df:
                out = out.where(df[c] != reason, reason)
    return out
