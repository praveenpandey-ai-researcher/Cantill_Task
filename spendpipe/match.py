"""Count once: duplicates inside a source and matches across systems.

Steps (all settings in the `matching` config section):

1. Exact duplicate rows inside any source (every field identical except the
   record id) -> later copies DUPLICATE.            [matching.exact_duplicates]
2. Cross-system matching, secondary (invoice tool) -> primary (ERP), in passes
   from strongest to weakest evidence. Each pass is ONE-TO-ONE: an ERP line
   can be claimed by only one tool line, so a second invoice for the same
   receipt is not silently swallowed.
     po_line_amount  same PO, same PO line, same net amount
     po_amount       same PO, same net amount
     po_text         same PO, similar item text, amount differs (partial /
                     combined invoice for goods already in the ERP)
     supplier_amount no PO at all: similar supplier + same net amount + same
                     quantity (+ dates within a window when both dates exist)
     po_text_any     same PO + similar text, ERP line may already be claimed
                     (many-to-one): extra split/repeated invoice for goods the
                     ERP already records
   A matched tool line gets DUPLICATE and `match_id` = the ERP record id.
3. Re-exported invoices among the tool lines still unmatched: same invoice-
   number core, supplier, net amount and item text, even if the raw formats
   differ -> later copies DUPLICATE.
   What is left is tool-only spend and stays included.

Only lines whose status counts take part: a cancelled invoice must not claim
an ERP line and leave the real invoice looking like extra spend.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from spendpipe.config import Config
from spendpipe.normalise import norm_invoice, norm_po, norm_po_line, norm_supplier, norm_text, squash

log = logging.getLogger(__name__)

DEFAULTS = {
    "exact_duplicates": True,
    "invoice_duplicates": True,
    "primary": None,            # source names; default = first primary / secondary
    "secondary": None,
    "po_pattern": None,
    "amount_tol_abs": 0.05,     # absolute tolerance in reporting currency
    "amount_tol_rel": 0.001,    # or 0.1% of the amount, whichever is larger
    "text_threshold": 75,       # 0-100 rapidfuzz token_set_ratio
    "supplier_threshold": 85,
    "date_window_days": 31,
    "fallback_only_without_po": True,
    "require_same_quantity": True,
    "passes": ["po_line_amount", "po_amount", "po_text", "supplier_amount"],
}


def _keys(df: pd.DataFrame, cfg_m: dict) -> pd.DataFrame:
    """Add normalised matching keys."""
    out = df.copy()
    pat = cfg_m.get("po_pattern")
    out["k_po"] = out["po_number"].map(lambda v: norm_po(v, pat))
    # the PO line is in its own column (ERP '10.0') or after a slash in the
    # PO reference (tool '4510008616/10.0'); otherwise unknown
    def po_line(line, po):
        if str(line).strip():
            return norm_po_line(line)
        return norm_po_line(po) if "/" in str(po) else ""
    out["k_pol"] = [po_line(a, b) for a, b in zip(out["po_line"], out["po_number"])]
    out["k_inv"] = out["invoice_number"].map(norm_invoice)
    out["k_sup"] = out["supplier"].map(norm_supplier)
    out["k_sup2"] = out["k_sup"].map(squash)
    out["k_txt"] = out["description"].map(norm_text)
    out["k_amt"] = out["spend_raw"].round(2)
    return out


def _amount_ok(a: pd.Series, b: pd.Series, cfg_m: dict) -> pd.Series:
    tol = np.maximum(cfg_m["amount_tol_abs"], cfg_m["amount_tol_rel"] * b.abs())
    return (a - b).abs() <= tol


def _supplier_sim(a: pd.Series, b: pd.Series) -> np.ndarray:
    return np.array([100.0 if (x and x == y) else fuzz.token_sort_ratio(x, y)
                     for x, y in zip(a, b)])


def _text_sim(a: pd.Series, b: pd.Series) -> np.ndarray:
    return np.array([fuzz.token_set_ratio(x, y) if (x and y) else 0.0 for x, y in zip(a, b)])


def _greedy_assign(cands: pd.DataFrame, claimed_p: set, claimed_s: set) -> list[tuple]:
    """Best score first; accept a pair only if both sides are still free."""
    out = []
    for s_id, p_id in zip(cands["s_id"], cands["p_id"]):
        if s_id in claimed_s or p_id in claimed_p:
            continue
        claimed_s.add(s_id)
        claimed_p.add(p_id)
        out.append((s_id, p_id))
    return out


def exact_duplicates(df: pd.DataFrame) -> pd.Series:
    """True for every row that repeats an earlier row of the same source in all
    fields except the record id."""
    ignore = {"record_id"}
    raw_cols = [c for c in df.columns if c.startswith("x_")] + [
        "amount", "date", "posting_date", "supplier", "supplier_id", "description", "item_code",
        "quantity", "status", "account", "account_desc", "cost_centre", "po_number", "po_line",
        "invoice_number", "currency", "fx_rate", "vat_rate", "document_type", "site"]
    cols = ["source"] + [c for c in raw_cols if c in df.columns and c not in ignore]
    return df.duplicated(cols, keep="first")


def match(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    cfg_m = {**DEFAULTS, **cfg.section("matching")}
    df = df.copy()
    df["dup_reason"] = ""
    df["match_id"] = ""
    df["match_pass"] = ""
    stats: dict = {}

    counting = df["status_reason"] == ""
    roles = {s["name"]: s["role"] for s in cfg.sources}
    sec = None
    if cfg_m.get("enabled"):
        sec = cfg_m["secondary"] or next((n for n, r in roles.items() if r == "secondary"), None)

    # 1. exact duplicate rows ------------------------------------------------------
    # Not for the secondary source of a cross-system match: two identical tool
    # lines can be two real receipts, so they go through matching first and
    # leftovers are handled by the re-export step (3).
    if cfg_m["exact_duplicates"]:
        ex = exact_duplicates(df) & counting & (df.source != sec)
        df.loc[ex, ["dup_reason", "match_pass"]] = ["DUPLICATE", "exact_row"]
        stats["exact_duplicate_rows"] = int(ex.sum())

    if not cfg_m.get("enabled"):
        return df, stats

    prim = cfg_m["primary"] or next(n for n, r in roles.items() if r == "primary")
    if sec is None:
        log.info("matching enabled but no secondary source; skipping cross-system match")
        return df, stats

    k = _keys(df, cfg_m)
    is_p = k.source == prim
    is_s = (k.source == sec) & counting & (df.dup_reason == "")

    # 2. cross-system passes ---------------------------------------------------------
    # (run BEFORE the re-export check: two identical tool lines can be two real
    # receipts, and each should be able to claim its own ERP line)
    P = k[is_p].rename(columns=lambda c: "p_" + c)
    S = k[is_s].rename(columns=lambda c: "s_" + c)
    P["p_id"], S["s_id"] = P.p_record_id, S.s_record_id
    claimed_p: set = set()
    claimed_s: set = set()
    matches: list[tuple] = []
    window = cfg_m["date_window_days"]

    for pas in cfg_m["passes"]:
        s_free = S[~S.s_id.isin(claimed_s)]
        # po_text_any may point at an ERP line that is already claimed: an extra
        # (split / repeated) invoice for goods the ERP already records.
        many_to_one = pas == "po_text_any"
        p_free = P if many_to_one else P[~P.p_id.isin(claimed_p)]
        if pas.startswith("po_"):
            c = s_free[s_free.s_k_po != ""].merge(p_free[p_free.p_k_po != ""],
                                                   left_on="s_k_po", right_on="p_k_po")
            if c.empty:
                continue
            amt = _amount_ok(c.s_k_amt, c.p_k_amt, cfg_m)
            if pas == "po_line_amount":
                c = c[amt & (c.s_k_pol != "") & (c.s_k_pol == c.p_k_pol)]
            elif pas == "po_amount":
                c = c[amt]
            # score = item-text similarity: picks the right ERP line when a PO
            # has several lines with the same amount
            c = c.assign(score=_text_sim(c.s_k_txt, c.p_k_txt))
            if pas in ("po_text", "po_text_any"):
                c = c[c.score >= cfg_m["text_threshold"]]
        elif pas == "supplier_amount":
            # Fallback only for tool lines WITHOUT a PO. A tool PO that is not in
            # the ERP marks tool-only spend: its supplier+amount look-alikes in
            # the ERP are coincidences (see DECISIONS.md, stage 3).
            if cfg_m.get("fallback_only_without_po", True):
                s_free = s_free[s_free.s_k_po == ""]
            # block on whole-currency amount (+/-1 for rounding), then filter
            s_ = s_free.assign(blk=np.floor(s_free.s_k_amt).astype("Int64"))
            p_ = pd.concat([p_free.assign(blk=np.floor(p_free.p_k_amt).astype("Int64") + d)
                            for d in (-1, 0, 1)])
            c = s_.dropna(subset=["blk"]).merge(p_.dropna(subset=["blk"]), on="blk")
            if c.empty:
                continue
            c = c[_amount_ok(c.s_k_amt, c.p_k_amt, cfg_m)]
            c = c.assign(sup=_supplier_sim(c.s_k_sup2, c.p_k_sup2))
            c = c[c.sup >= cfg_m["supplier_threshold"]]
            # same quantity when both sides have one (true duplicates: 97-100%)
            if cfg_m.get("require_same_quantity", True):
                qs = pd.to_numeric(c.s_quantity, errors="coerce")
                qp = pd.to_numeric(c.p_quantity, errors="coerce")
                c = c[qs.isna() | qp.isna() | ((qs - qp).abs() < 1e-6)]
            # dates within the window when BOTH exist (most are blank)
            both = c.s_date_parsed.notna() & c.p_date_parsed.notna()
            gap = (c.s_date_parsed - c.p_date_parsed).abs().dt.days
            c = c[~both | (gap <= window)]
            c = c.assign(score=_text_sim(c.s_k_txt, c.p_k_txt) + c.sup / 10
                         - gap.fillna(window).loc[c.index] / max(window, 1))
        else:
            log.warning("unknown matching pass %s", pas)
            continue
        c = c.sort_values("score", ascending=False, kind="stable")
        if many_to_one:   # best ERP line per tool line, ERP lines may repeat
            c = c.drop_duplicates("s_id")
            got = list(zip(c.s_id, c.p_id))
            claimed_s.update(c.s_id)
        else:
            got = _greedy_assign(c, claimed_p, claimed_s)
        matches += [(s, p, pas) for s, p in got]
        stats[f"pass_{pas}"] = len(got)
        log.info("matching pass %-16s matched %6d tool lines", pas, len(got))

    if matches:
        m = pd.DataFrame(matches, columns=["s_id", "p_id", "pass"]).set_index("s_id")
        rows = df.record_id.isin(m.index)
        df.loc[rows, "dup_reason"] = "DUPLICATE"
        df.loc[rows, "match_id"] = df.loc[rows, "record_id"].map(m.p_id)
        df.loc[rows, "match_pass"] = df.loc[rows, "record_id"].map(m["pass"])

    # 3. re-exported invoices among the remaining tool-only lines ----------------
    # Same invoice-number core + supplier + net amount + item text, even when
    # the raw invoice/PO/date formats differ -> later copy is DUPLICATE.
    if cfg_m["invoice_duplicates"]:
        left = is_s & (df.dup_reason == "") & (k.k_inv != "")
        sk = k[left]
        key = ["k_inv", "k_sup2", "k_amt", "k_txt"]
        rep = sk.duplicated(key, keep="first")
        first_of = sk.groupby(key).record_id.transform("first")
        idx = rep[rep].index
        df.loc[idx, "dup_reason"] = "DUPLICATE"
        df.loc[idx, "match_pass"] = "reexported_invoice"
        df.loc[idx, "match_id"] = first_of[idx]
        stats["reexported_invoice_lines"] = int(len(idx))

    sec_all = df.source == sec
    stats["secondary_lines"] = int(sec_all.sum())
    stats["secondary_counting"] = int((sec_all & counting).sum())
    stats["secondary_matched_to_primary"] = len(matches)
    stats["secondary_only_included"] = int((sec_all & counting & (df.dup_reason == "")).sum())
    stats["secondary_only_spend"] = round(float(
        df.loc[sec_all & counting & (df.dup_reason == ""), "spend_raw"].sum()), 2)
    return df, stats
