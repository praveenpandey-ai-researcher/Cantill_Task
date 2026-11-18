"""Clean amounts, VAT, currency and dates.

The small parse_* functions work on single values and are unit-tested.
`clean()` applies them to the whole canonical frame and adds:

    amount_num     parsed number from the amount column (NaN if unparsable)
    vat_rate_num   VAT fraction used (0 for net sources)
    net_amount     amount without VAT, in the line currency
    fx_rate_used   units of line currency per 1 reporting-currency unit
    fx_source      same_ccy | given | table:<month> | table_nearest:<month> | missing
    spend_raw      net amount in the reporting currency (before exclusions)
    date_parsed    parsed main date (NaT if blank/unparsable)
    date_flag      ok | blank | unparsable | implausible
    dq_flags       ';'-joined data-quality notes for the line

Nothing here raises on bad data: bad values become NaN/NaT plus a flag.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil import parser as du_parser

from spendpipe.config import Config

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Amounts
# --------------------------------------------------------------------------- #
_CURRENCY_JUNK = re.compile(r"[^\d,.\-()+eE]")


def parse_amount(value, decimal: str = "auto") -> float:
    """Parse a money string to float. Returns NaN if it cannot be parsed.

    Handles: '1,234.56'  '1.234,56'  '1 234,56'  '(123.45)'  '123.45-'  'R 1,000'
    decimal='auto' decides per value:
      * both ',' and '.' present -> the LAST one is the decimal separator
      * only ',' present -> decimal if followed by 1-2 digits at the end, else thousands
    decimal='.' or ',' forces the convention.
    """
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = str(value).strip()
    if not s:
        return np.nan
    negative = False
    if s.startswith("(") and s.endswith(")"):          # accounting negative
        negative, s = True, s[1:-1]
    if s.endswith("-"):                                # SAP trailing minus
        negative, s = True, s[:-1]
    s = _CURRENCY_JUNK.sub("", s)                     # drop symbols, spaces, letters
    if s.startswith("-"):
        negative, s = (not negative), s[1:]
    s = s.lstrip("+")
    if not s:
        return np.nan

    if decimal == "auto":
        if "," in s and "." in s:
            decimal = "," if s.rfind(",") > s.rfind(".") else "."
        elif "," in s:
            decimal = "," if re.search(r",\d{1,2}$", s) else "."
        else:
            decimal = "."
    thousands = "." if decimal == "," else ","
    s = s.replace(thousands, "")
    if decimal == ",":
        s = s.replace(",", ".")
    try:
        x = float(s)
    except ValueError:
        return np.nan
    return -x if negative else x


def parse_rate(value, default: float | None = None) -> float:
    """VAT rate to a fraction: '15', '15%', '0.15', 15 -> 0.15. '0' -> 0.0.
    Blank/unparsable -> `default` (NaN if no default)."""
    x = parse_amount(value)
    if np.isnan(x):
        return np.nan if default is None else float(default)
    if isinstance(value, str) and "%" in value:
        return x / 100.0
    return x / 100.0 if x >= 1 else x       # '15' or '1' are percents; VAT is never 100%


def remove_vat(gross: float, rate: float) -> float:
    """Net = gross / (1 + rate). A rate of 0 leaves the amount unchanged."""
    if np.isnan(gross) or np.isnan(rate):
        return np.nan
    return gross / (1.0 + rate)


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
# Month names in other languages -> English, so one parser handles them all.
_MONTHS = {
    # Spanish / Portuguese / Italian / French / German / Dutch (common abbreviations)
    "ene": "jan", "enero": "jan", "janeiro": "jan", "gennaio": "jan", "janv": "jan", "janvier": "jan", "januar": "jan", "jän": "jan",
    "febrero": "feb", "fev": "feb", "fevereiro": "feb", "febbraio": "feb", "févr": "feb", "fevr": "feb", "février": "feb", "februar": "feb",
    "marzo": "mar", "março": "mar", "marco": "mar", "mars": "mar", "märz": "mar", "maerz": "mar", "mrt": "mar",
    "abr": "apr", "abril": "apr", "aprile": "apr", "avr": "apr", "avril": "apr",
    "mayo": "may", "maio": "may", "mag": "may", "maggio": "may", "mai": "may", "mei": "may",
    "junio": "jun", "junho": "jun", "giu": "jun", "giugno": "jun", "juin": "jun", "juni": "jun",
    "julio": "jul", "julho": "jul", "lug": "jul", "luglio": "jul", "juil": "jul", "juillet": "jul", "juli": "jul",
    "ago": "aug", "agosto": "aug", "août": "aug", "aout": "aug",
    "set": "sep", "septiembre": "sep", "setembro": "sep", "settembre": "sep", "sept": "sep", "septembre": "sep",
    "octubre": "oct", "outubro": "oct", "ott": "oct", "ottobre": "oct", "octobre": "oct", "okt": "oct", "oktober": "oct",
    "noviembre": "nov", "novembro": "nov", "novembre": "nov",
    "dic": "dec", "diciembre": "dec", "dez": "dec", "dezembro": "dec", "dicembre": "dec", "déc": "dec", "decembre": "dec", "décembre": "dec", "dezember": "dec",
}
_MONTH_RE = re.compile(r"[A-Za-zÀ-ÿ]+")


def _english_months(s: str) -> str:
    return _MONTH_RE.sub(lambda m: _MONTHS.get(m.group(0).lower().rstrip("."), m.group(0)), s)


def parse_date(value, formats: list[str] | None = None, dayfirst: bool = True):
    """Parse one date string. Returns a `datetime` or None.

    Order: explicit `formats` (from config) -> ISO -> Excel serial number ->
    flexible parser (dayfirst for ambiguous 03/04/2023) with foreign month names.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "nat", "none", "null", "0", "00000000", "-"}:
        return None
    for fmt in (formats or []):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):            # ISO, maybe with a time part
        try:
            return datetime.fromisoformat(s[:10])
        except ValueError:
            return None                                # e.g. 2023-02-30: impossible date
    if re.fullmatch(r"\d{8}", s):                     # 20240131 (SAP style)
        try:
            return datetime.strptime(s, "%Y%m%d")
        except ValueError:
            return None
    if re.fullmatch(r"\d{5}(\.\d+)?", s):             # Excel serial, e.g. 45321
        n = float(s)
        if 20000 < n < 80000:
            return datetime(1899, 12, 30) + pd.Timedelta(days=n)
        return None
    try:
        return du_parser.parse(_english_months(s), dayfirst=dayfirst, fuzzy=False)
    except (ValueError, OverflowError, TypeError):
        return None


def parse_dates(series: pd.Series, formats=None, dayfirst=True) -> pd.Series:
    """Vectorised parse_date: parses each *unique* value once (fast)."""
    uniq = series.dropna().unique()
    lookup = {v: parse_date(v, formats, dayfirst) for v in uniq}
    return pd.to_datetime(series.map(lookup), errors="coerce")


# --------------------------------------------------------------------------- #
# FX
# --------------------------------------------------------------------------- #
class FxTable:
    """Monthly rates: units of currency per 1 reporting-currency unit."""

    def __init__(self, df: pd.DataFrame | None):
        self.rates: dict[str, pd.Series] = {}
        if df is None or df.empty:
            return
        for ccy, grp in df.groupby("currency"):
            s = grp.set_index("month")["rate"].sort_index()
            self.rates[str(ccy).upper()] = s

    @classmethod
    def from_config(cls, cfg: Config) -> "FxTable":
        fx = cfg.section("fx")
        path = cfg.file(fx.get("table_file"))
        if not path or not path.exists():
            return cls(None)
        from spendpipe.ingest import read_table
        raw = read_table(path)
        cols = {k: fx.get(f"{k}_column", d) for k, d in
                [("currency", "currency"), ("month", "month"), ("rate", "units_per_usd")]}
        df = pd.DataFrame({
            "currency": raw[cols["currency"]].str.strip().str.upper(),
            "month": raw[cols["month"]].str.strip().str[:7],
            "rate": raw[cols["rate"]].map(parse_amount),
        })
        df = df[(df.rate > 0) & df.month.str.match(r"^\d{4}-\d{2}$")]
        log.info("FX table: %d currencies, %d monthly rates", df.currency.nunique(), len(df))
        return cls(df)

    def lookup(self, ccy: str, month: str | None) -> tuple[float, str]:
        """Rate for the month; else the nearest month available. (rate, how)."""
        s = self.rates.get(ccy)
        if s is None or s.empty:
            return np.nan, "missing"
        if month and month in s.index:
            return float(s[month]), f"table:{month}"
        if not month:
            return float(s.iloc[-1]), "table_nearest:latest"
        # nearest month by distance in months
        def m2i(m):
            return int(m[:4]) * 12 + int(m[5:7])
        target = m2i(month)
        idx = min(s.index, key=lambda m: abs(m2i(m) - target))
        return float(s[idx]), f"table_nearest:{idx}"


def fx_rate_for(ccy: str, given_rate: float, month: str | None, reporting: str,
                table: FxTable, max_deviation: float | None = None) -> tuple[float, str]:
    """Pick the FX rate for one line (units of `ccy` per 1 `reporting`).

    1. Line already in reporting currency -> 1.0, ignoring any given rate
       (B has USD lines carrying an MXN-sized rate; the brief says USD is USD).
    2. A positive given rate on the line -> use it, unless it is more than
       `max_deviation` times away from the table rate for that month (B has EUR
       lines carrying an MXN-sized rate, 19x off) -> then the table wins.
    3. Otherwise the monthly table (exact month, else nearest month).
    """
    if ccy == reporting:
        return 1.0, "same_ccy"
    if given_rate is not None and not np.isnan(given_rate) and given_rate > 0:
        if max_deviation:
            ref, how = table.lookup(ccy, month)
            if not np.isnan(ref):
                ratio = given_rate / ref
                if ratio > max_deviation or ratio < 1.0 / max_deviation:
                    return ref, "given_rejected>" + how
        return float(given_rate), "given"
    return table.lookup(ccy, month)


def to_reporting(amount: float, rate: float, rate_meaning: str = "units_per_reporting") -> float:
    """Convert a line-currency amount with a rate.
    units_per_reporting: rate = units of line ccy per 1 reporting unit -> DIVIDE.
    reporting_per_unit:  rate = reporting units per 1 line unit       -> MULTIPLY."""
    if np.isnan(amount) or np.isnan(rate) or rate == 0:
        return np.nan
    return amount / rate if rate_meaning == "units_per_reporting" else amount * rate


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
def clean_text(s: str) -> str:
    """Trim, collapse whitespace, drop trailing ' -' separators (JDE exports)."""
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    s = re.sub(r"(\s*-\s*)+$", "", s).strip()
    return s


# --------------------------------------------------------------------------- #
# Whole-frame cleaning
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    reporting = cfg.reporting_currency.upper()
    fxcfg = cfg.section("fx")
    rate_meaning = fxcfg.get("rate_meaning", "units_per_reporting")
    table = FxTable.from_config(cfg)
    dcfg = cfg.section("dates")
    min_d = pd.Timestamp(dcfg.get("min", "2000-01-01"))
    max_d = pd.Timestamp(dcfg.get("max", "2035-12-31"))
    max_gap = dcfg.get("max_gap_days")      # vs posting_date, optional

    parts = []
    for src in cfg.sources:
        part = df[df.source == src["name"]].copy()
        if part.empty:
            continue
        acfg = src.get("amount") or {}

        # --- amount and VAT ---------------------------------------------------
        decimal = src.get("decimal", "auto")
        part["amount_num"] = part["amount"].map(lambda v: parse_amount(v, decimal))
        if acfg.get("gross"):
            default = acfg.get("default_vat_rate")
            part["vat_rate_num"] = part["vat_rate"].map(lambda v: parse_rate(v, default))
        else:
            part["vat_rate_num"] = 0.0
        part["net_amount"] = [remove_vat(a, r) for a, r in zip(part.amount_num, part.vat_rate_num)]

        # --- dates --------------------------------------------------------------
        fmts, dayfirst = src.get("date_formats") or [], src.get("dayfirst", True)
        part["date_parsed"] = parse_dates(part["date"], fmts, dayfirst)
        post = parse_dates(part["posting_date"], fmts, dayfirst)
        part["posting_date_parsed"] = post
        blank = part["date"].str.strip() == ""
        unparsable = ~blank & part["date_parsed"].isna()
        bad_range = part["date_parsed"].notna() & ((part["date_parsed"] < min_d) | (part["date_parsed"] > max_d))
        if max_gap:
            gap = (part["date_parsed"] - post).abs().dt.days
            bad_range |= gap.notna() & (gap > int(max_gap))
        part["date_flag"] = np.select([blank, unparsable, bad_range],
                                      ["blank", "unparsable", "implausible"], "ok")

        # --- currency -----------------------------------------------------------
        part["currency"] = part["currency"].str.strip().str.upper()
        part.loc[part["currency"] == "", "currency"] = reporting
        aliases = {k.upper(): v.upper() for k, v in (fxcfg.get("aliases") or {}).items()}
        if aliases:
            part["currency"] = part["currency"].replace(aliases)
        given = part["fx_rate"].map(parse_amount)
        # FX month: main date if usable, else posting date, else unknown.
        good = part["date_flag"] == "ok"
        month_date = part["date_parsed"].where(good, post)
        month = month_date.dt.strftime("%Y-%m").where(month_date.notna(), None)
        max_dev = fxcfg.get("max_rate_deviation")
        res = [fx_rate_for(c, g, m, reporting, table, max_dev)
               for c, g, m in zip(part["currency"], given, month)]
        part["fx_rate_used"] = [r for r, _ in res]
        part["fx_source"] = [h for _, h in res]
        part["spend_raw"] = [to_reporting(a, r, rate_meaning)
                             for a, r in zip(part.net_amount, part.fx_rate_used)]

        # --- text ---------------------------------------------------------------
        for col in ("description", "supplier"):
            part[col] = part[col].map(clean_text)
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    flags = pd.Series("", index=out.index)
    def add(mask, label):
        nonlocal flags
        flags = flags.where(~mask, flags + label + ";")
    add(out.amount_num.isna(), "AMOUNT_UNPARSABLE")
    add(out.date_flag == "unparsable", "DATE_UNPARSABLE")
    add(out.date_flag == "implausible", "DATE_IMPLAUSIBLE")
    add(out.fx_source == "missing", "FX_MISSING")
    add(out.fx_source.str.startswith("given_rejected"), "FX_GIVEN_RATE_REJECTED")
    add(out.fx_source.str.startswith("table_nearest"), "FX_NEAREST_MONTH")
    add(out.description == "", "DESC_BLANK")
    out["dq_flags"] = flags.str.rstrip(";")
    return out


def summarise(df: pd.DataFrame, reporting: str) -> dict:
    """Numbers for the log and the data-quality report."""
    s: dict = {"rows": int(len(df))}
    for src, g in df.groupby("source"):
        s[src] = {
            "rows": int(len(g)),
            "amount_unparsable": int(g.amount_num.isna().sum()),
            "negative_amounts": int((g.amount_num < 0).sum()),
            "date_flags": {k: int(v) for k, v in g.date_flag.value_counts().items()},
            "fx_source": {k: int(v) for k, v in g.fx_source.str.split(":").str[0].value_counts().items()},
            "blank_description": int((g.description == "").sum()),
            f"spend_raw_total_{reporting}": round(float(g.spend_raw.sum()), 2),
        }
    return s
