"""Load and validate a company config (YAML).

Everything company-specific lives in the config: file names, column names,
formats, status words, exclusion rules, thresholds. The code only knows the
*canonical* field names below (amount, date, supplier, ...), never a company's
column names or category names.

File paths in the config are relative to `data_dir`, which itself is relative
to the config file's folder. `--data-dir` on the command line overrides it, so
the blind-test files (A2, B2) can be run with the same config.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

# Canonical fields a source can map its columns onto. A source maps a subset.
# A value in the config can be one column name or a list of names (joined with
# a space, e.g. material text + service text -> description).
CANONICAL_FIELDS = {
    "record_id",      # unique id of the line in its source file (required)
    "amount",         # money value (required)
    "date",           # main date used for FX month and match windows
    "posting_date",   # secondary date: fallback when `date` is blank/implausible
    "supplier",       # supplier name as typed
    "supplier_id",    # vendor id / number
    "description",    # free text describing what was bought
    "item_code",      # material / item number
    "quantity",
    "uom",
    "status",
    "account",        # GL account code
    "account_desc",   # GL account description
    "cost_centre",
    "po_number",
    "po_line",
    "invoice_number",
    "currency",
    "fx_rate",
    "vat_rate",
    "document_type",
    "supplier_type",
    "site",
}

# Defaults merged under every config, so configs only state what differs.
DEFAULTS: dict[str, Any] = {
    "reporting_currency": "USD",
    "data_dir": ".",
    "sources": [],
    "taxonomy": {"file": None, "level_columns": None},
    "labels": None,
    "fx": {
        "table_file": None,           # monthly fallback table
        "rate_meaning": "units_per_reporting",  # rate = units of line ccy per 1 reporting unit -> divide
    },
    "matching": {"enabled": False},
    "exclusions": {},
    "dedup": {
        "supplier": {"threshold": 90},
        "item": {"threshold": 0.9},
    },
    # Classifier defaults live in ONE place: spendpipe/classify.py (DEFAULTS).
    "classify": {},
}

SOURCE_DEFAULTS: dict[str, Any] = {
    "role": "primary",        # primary records win over secondary in count-once
    "id_prefix": "",          # e.g. "ERP:" -> record_id "ERP:A000002"
    "columns": {},
    "date_formats": [],       # tried in order, then a flexible parser
    "dayfirst": True,         # for ambiguous dates like 03/04/2023
    "amount": {
        "gross": False,       # True -> remove VAT
        "default_vat_rate": None,
    },
    "currency": {"fixed": None},   # fixed currency if there is no column
    "status": {},             # maps status values -> include / exclusion code
    "encoding": "utf-8",
}


@dataclass
class Config:
    company: str
    path: Path
    data_dir: Path
    raw: dict[str, Any] = field(repr=False)

    # Convenience accessors -------------------------------------------------
    @property
    def sources(self) -> list[dict[str, Any]]:
        return self.raw["sources"]

    @property
    def reporting_currency(self) -> str:
        return self.raw["reporting_currency"]

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name) or {}

    def file(self, rel: str | None) -> Path | None:
        """Resolve a data file name against data_dir."""
        if not rel:
            return None
        p = Path(rel)
        return p if p.is_absolute() else (self.data_dir / p)


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, data_dir: str | Path | None = None) -> Config:
    path = Path(path).resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}

    raw = _deep_merge(DEFAULTS, user)
    raw["sources"] = [_deep_merge(SOURCE_DEFAULTS, s) for s in user.get("sources", [])]

    if data_dir is not None:
        ddir = Path(data_dir).resolve()
    else:
        ddir = (path.parent / raw["data_dir"]).resolve()

    cfg = Config(company=str(raw.get("company", path.stem)), path=path, data_dir=ddir, raw=raw)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Fail early with a clear message on config mistakes; warn on soft issues."""
    problems: list[str] = []
    if not cfg.sources:
        problems.append("config has no `sources`")

    names = set()
    for i, src in enumerate(cfg.sources):
        tag = f"sources[{i}] ({src.get('name', '?')})"
        if not src.get("name"):
            problems.append(f"{tag}: missing `name`")
        if src.get("name") in names:
            problems.append(f"{tag}: duplicate source name")
        names.add(src.get("name"))
        if not src.get("file"):
            problems.append(f"{tag}: missing `file`")
        elif not cfg.file(src["file"]).exists():
            problems.append(f"{tag}: file not found: {cfg.file(src['file'])}")
        cols = src.get("columns", {})
        for req in ("record_id", "amount"):
            if req not in cols:
                problems.append(f"{tag}: `columns.{req}` is required")
        unknown = set(cols) - CANONICAL_FIELDS
        if unknown:
            problems.append(f"{tag}: unknown canonical field(s) {sorted(unknown)}; "
                            f"allowed: {sorted(CANONICAL_FIELDS)}")
        if src.get("role") not in ("primary", "secondary"):
            problems.append(f"{tag}: role must be 'primary' or 'secondary'")

    tax_file = cfg.section("taxonomy").get("file")
    if not tax_file:
        problems.append("`taxonomy.file` is required")
    elif not cfg.file(tax_file).exists():
        problems.append(f"taxonomy file not found: {cfg.file(tax_file)}")

    fx_file = cfg.section("fx").get("table_file")
    if fx_file and not cfg.file(fx_file).exists():
        log.warning("FX table %s not found; lines without a rate will be flagged", cfg.file(fx_file))

    if problems:
        raise ConfigError("Invalid config %s:\n  - %s" % (cfg.path, "\n  - ".join(problems)))
