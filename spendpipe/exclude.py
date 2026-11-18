"""Exclusion rules and reason codes.

Every line ends with `include` (1/0) and `exclusion_reason`:
    CANCELLED        cancelled invoice or line
    NOT_APPROVED     rejected, on hold, submitted, disputed...
    NON_PROCUREMENT  taxes, intercompany, finance... (stage 4)
    DUPLICATE        already counted in another record (stage 3, match.py)

When several apply, the first in PRECEDENCE wins. The record's own nature comes
first: a cancelled invoice is CANCELLED even if it also matches the ERP, and a
duplicated tax line is NON_PROCUREMENT, because DUPLICATE means "already
counted in another record" and the original tax line is not counted either.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from spendpipe.config import Config

log = logging.getLogger(__name__)

PRECEDENCE = ["CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"]

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


# --------------------------------------------------------------------------- #
# NON_PROCUREMENT
# --------------------------------------------------------------------------- #
def _compile(patterns) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in (patterns or [])]


def load_lexicon(cfg: Config) -> dict:
    """Read the shared lexicon YAML (path relative to the config file)."""
    import yaml
    npc = cfg.section("exclusions").get("non_procurement") or {}
    path = npc.get("lexicon")
    lex: dict = {"groups": {}, "supplier_patterns": []}
    if path:
        full = (cfg.path.parent / path).resolve()
        if full.exists():
            with open(full, encoding="utf-8") as fh:
                lex = yaml.safe_load(fh) or lex
        else:
            log.warning("lexicon %s not found; only config rules are used", full)
    for g in npc.get("disable_groups", []):
        lex["groups"].pop(g, None)
    for g, pats in (npc.get("extra_patterns") or {}).items():
        lex["groups"].setdefault(g, {"patterns": []})["patterns"] += pats
    return lex


class NonProcurementRules:
    """Decides, per line, whether it is NON_PROCUREMENT and which rule fired."""

    def __init__(self, cfg: Config):
        npc = cfg.section("exclusions").get("non_procurement") or {}
        lex = load_lexicon(cfg)
        self.text_fields = npc.get("text_fields", ["description"])
        self.accounts = {str(a).strip().upper() for a in npc.get("accounts", [])}
        self.account_desc = _compile(npc.get("account_desc_patterns"))
        self.supplier_ids = {str(s).strip().upper() for s in npc.get("supplier_ids", [])}
        self.supplier_pats = _compile(lex.get("supplier_patterns", []) + npc.get("supplier_patterns", []))
        self.groups = {name: (_compile(g.get("patterns")), _compile(g.get("unless")))
                       for name, g in (lex.get("groups") or {}).items()}
        log.info("NON_PROCUREMENT rules: %d lexicon groups, %d account patterns, %d accounts, "
                 "%d supplier patterns", len(self.groups), len(self.account_desc),
                 len(self.accounts), len(self.supplier_pats))

    def text_rule(self, text: str) -> str:
        """Lexicon group that matches `text`, or ''."""
        if not text:
            return ""
        for name, (pats, unless) in self.groups.items():
            if any(p.search(text) for p in pats) and not any(u.search(text) for u in unless):
                return name
        return ""

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Return the rule that fired per line ('' = procurement spend)."""
        rule = pd.Series("", index=df.index)

        def first(mask, label):
            nonlocal rule
            rule = rule.where(~(mask & (rule == "")), label)

        if self.accounts:
            first(df["account"].str.strip().str.upper().isin(self.accounts), "account")
        if self.account_desc:
            u = df["account_desc"].unique()
            hit = {v: any(p.search(v) for p in self.account_desc) for v in u}
            first(df["account_desc"].map(hit), "account_desc")
        if self.supplier_ids:
            first(df["supplier_id"].str.strip().str.upper().isin(self.supplier_ids), "supplier_id")
        if self.supplier_pats:
            u = df["supplier"].unique()
            hit = {v: any(p.search(v) for p in self.supplier_pats) for v in u}
            first(df["supplier"].map(hit), "supplier_pattern")
        for f in self.text_fields:
            if f not in df:
                continue
            u = df[f].unique()
            lab = {v: self.text_rule(v) for v in u}
            got = df[f].map(lab)                  # lexicon group name or ''
            rule = rule.where(~((rule == "") & (got != "")), "text:" + got)
        return rule


def apply_non_procurement(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    rules = NonProcurementRules(cfg)
    df["np_rule"] = rules.apply(df)
    df["np_reason"] = (df["np_rule"] != "").map({True: "NON_PROCUREMENT", False: ""})
    return df


def finalise(df: pd.DataFrame) -> pd.DataFrame:
    """One exclusion_reason per line by precedence; include = 1 when none."""
    df = df.copy()
    df["exclusion_reason"] = combine_reasons(df, ["status_reason", "dup_reason", "np_reason"])
    df["include"] = (df["exclusion_reason"] == "").astype(int)
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
