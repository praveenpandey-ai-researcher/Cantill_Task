"""Read each source file and map its columns onto the canonical schema.

Output: one DataFrame with one row per input record and these columns:
    record_id, source, role, <canonical fields...>
All values are kept as *strings* here; parsing happens in clean.py.

Robustness rules:
  * every cell is read as text (no silent type guessing, IDs keep leading zeros)
  * a mapped column that is missing from the file -> warning + blank column
  * unknown extra columns in the file are ignored (but logged)
  * a list of columns is joined with a space (e.g. material text + service text)
"""
from __future__ import annotations

import logging

import pandas as pd

from spendpipe.config import CANONICAL_FIELDS, Config

log = logging.getLogger(__name__)


def sniff_separator(path, encoding: str = "utf-8") -> str:
    """Guess the separator from the header line: the most frequent of , ; | tab."""
    with open(path, encoding=encoding, errors="replace") as fh:
        header = fh.readline()
    counts = {s: header.count(s) for s in [",", ";", "|", "\t"]}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ","


def read_table(path, encoding: str = "utf-8", sep: str | None = None) -> pd.DataFrame:
    """Read a CSV as all-text. Falls back to latin-1 if the encoding is wrong,
    and sniffs the separator when none is given (`;` files exist)."""
    kwargs = dict(dtype=str, keep_default_na=False, na_values=[],
                  sep=sep or sniff_separator(path, encoding))
    try:
        df = pd.read_csv(path, encoding=encoding, **kwargs)
    except UnicodeDecodeError:
        log.warning("%s is not %s; re-reading as latin-1", path, encoding)
        df = pd.read_csv(path, encoding="latin-1", **kwargs)
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df


def _pick(df: pd.DataFrame, spec, source: str, field: str) -> pd.Series:
    """Return the column(s) named by `spec` as one text Series."""
    cols = spec if isinstance(spec, list) else [spec]
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        log.warning("[%s] column(s) %s for field '%s' not in file; treated as blank",
                    source, missing, field)
    if not present:
        return pd.Series("", index=df.index, dtype=object)
    out = df[present[0]].fillna("").astype(str).str.strip()
    for c in present[1:]:
        nxt = df[c].fillna("").astype(str).str.strip()
        out = (out + " " + nxt).str.strip()
    return out


def ingest_source(cfg: Config, src: dict) -> pd.DataFrame:
    path = cfg.file(src["file"])
    raw = read_table(path, encoding=src.get("encoding", "utf-8"), sep=src.get("sep"))
    name = src["name"]
    mapped = {v for spec in src["columns"].values() for v in (spec if isinstance(spec, list) else [spec])}
    unused = [c for c in raw.columns if c not in mapped]
    log.info("[%s] %s: %d rows, %d columns (%d not mapped)", name, path.name, len(raw),
             raw.shape[1], len(unused))

    out = pd.DataFrame(index=raw.index)
    for field in sorted(CANONICAL_FIELDS):
        spec = src["columns"].get(field)
        out[field] = _pick(raw, spec, name, field) if spec else ""

    out["record_id"] = src.get("id_prefix", "") + out["record_id"]
    out.insert(0, "source", name)
    out.insert(1, "role", src["role"])
    # `currency.fixed` means "the amount column is always in this currency"
    # (A's ERP: value_zar is ZAR even where document_currency says EUR).
    # The original column value stays available as `currency_doc`.
    fixed = (src.get("currency") or {}).get("fixed")
    if fixed:
        out["currency_doc"] = out["currency"]
        out["currency"] = fixed
    # Unmapped columns are kept under an `x_` prefix: later stages (e.g. the
    # classifier's fallback signals) may use them by name from the config.
    for c in unused:
        out["x_" + c] = raw[c].fillna("").astype(str)

    dup_ids = out["record_id"].duplicated().sum()
    if dup_ids:
        log.warning("[%s] %d duplicated record ids", name, dup_ids)
    return out


def ingest(cfg: Config) -> pd.DataFrame:
    frames = [ingest_source(cfg, s) for s in cfg.sources]
    df = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    return df
