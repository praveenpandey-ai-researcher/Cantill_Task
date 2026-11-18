"""Unit tests for the money-critical cleaning functions (VAT, FX, amounts, dates)."""
import math
from datetime import datetime

import pandas as pd
import pytest

from spendpipe.clean import (FxTable, clean_text, fx_rate_for, parse_amount, parse_date,
                             parse_rate, remove_vat, to_reporting)


# ---------------------------------------------------------------- amounts
@pytest.mark.parametrize("raw,expected", [
    ("1234.56", 1234.56),
    ("1,234.56", 1234.56),          # thousands comma (A tool export)
    ("1.234,56", 1234.56),          # European
    ("1 234,56", 1234.56),          # space thousands
    ("12,5", 12.5),                 # decimal comma, no thousands
    ("1,234", 1234.0),              # comma followed by 3 digits = thousands
    ("-13827.64", -13827.64),
    ("(500.00)", -500.0),           # accounting negative
    ("500.00-", -500.0),            # SAP trailing minus
    ("R 1,000.00", 1000.0),         # currency symbol
    ("", math.nan),
    ("abc", math.nan),
])
def test_parse_amount(raw, expected):
    got = parse_amount(raw)
    assert (math.isnan(got) and math.isnan(expected)) or got == pytest.approx(expected)


def test_parse_amount_forced_decimal():
    assert parse_amount("1.234", decimal=",") == 1234.0
    assert parse_amount("1,234", decimal=",") == pytest.approx(1.234)


# ---------------------------------------------------------------- VAT
@pytest.mark.parametrize("raw,expected", [
    ("15", 0.15), ("15%", 0.15), ("0.15", 0.15), ("0", 0.0), ("14.5%", 0.145),
])
def test_parse_rate_formats(raw, expected):
    assert parse_rate(raw, default=0.15) == pytest.approx(expected)


def test_blank_vat_uses_default_but_zero_stays_zero():
    assert parse_rate("", default=0.15) == pytest.approx(0.15)
    assert parse_rate("0", default=0.15) == 0.0        # zero-rated is NOT blank


def test_remove_vat():
    assert remove_vat(115.0, 0.15) == pytest.approx(100.0)
    assert remove_vat(100.0, 0.0) == 100.0
    assert remove_vat(-115.0, 0.15) == pytest.approx(-100.0)   # credit notes too
    assert math.isnan(remove_vat(math.nan, 0.15))


# ---------------------------------------------------------------- FX
@pytest.fixture
def table():
    return FxTable(pd.DataFrame({
        "currency": ["EUR", "EUR", "CAD"],
        "month": ["2024-01", "2024-03", "2024-02"],
        "rate": [0.90, 0.92, 1.35],
    }))


def test_fx_direction_is_divide(table):
    # 0.9 EUR per USD -> 90 EUR is 100 USD
    rate, how = fx_rate_for("EUR", 0.9, "2024-01", "USD", table)
    assert how == "given"
    assert to_reporting(90.0, rate) == pytest.approx(100.0)
    # the opposite convention multiplies
    assert to_reporting(90.0, 2.0, "reporting_per_unit") == pytest.approx(180.0)


def test_usd_line_ignores_its_rate(table):
    # B trap: USD lines carrying an MXN-sized rate must stay as they are
    rate, how = fx_rate_for("USD", 17.35, "2024-01", "USD", table)
    assert (rate, how) == (1.0, "same_ccy")
    assert to_reporting(500.0, rate) == 500.0


def test_wildly_wrong_given_rate_is_replaced_by_table(table):
    # EUR line carrying an MXN-sized rate (~17.6 vs 0.90): 19x off -> table
    rate, how = fx_rate_for("EUR", 17.6, "2024-01", "USD", table, max_deviation=5)
    assert rate == 0.90 and how.startswith("given_rejected")
    # a plausible given rate is kept
    assert fx_rate_for("EUR", 0.95, "2024-01", "USD", table, max_deviation=5) == (0.95, "given")


def test_missing_rate_uses_table_month_then_nearest(table):
    assert fx_rate_for("CAD", math.nan, "2024-02", "USD", table) == (1.35, "table:2024-02")
    rate, how = fx_rate_for("EUR", math.nan, "2024-02", "USD", table)
    assert how.startswith("table_nearest") and rate in (0.90, 0.92)
    assert fx_rate_for("EUR", 0.0, "2024-03", "USD", table)[1] == "table:2024-03"  # 0 rate = missing


def test_unknown_currency_is_flagged_not_crashing(table):
    rate, how = fx_rate_for("XYZ", math.nan, "2024-01", "USD", table)
    assert how == "missing" and math.isnan(rate)
    assert math.isnan(to_reporting(100.0, rate))


# ---------------------------------------------------------------- dates
@pytest.mark.parametrize("raw,expected", [
    ("2023-03-27", datetime(2023, 3, 27)),
    ("27/03/2023", datetime(2023, 3, 27)),
    ("03/04/2023", datetime(2023, 4, 3)),      # ambiguous -> day first
    ("27-Mar-24", datetime(2024, 3, 27)),
    ("20240131", datetime(2024, 1, 31)),       # SAP compact
    ("45292", datetime(2024, 1, 1)),           # Excel serial
    ("5 mayo 2024", datetime(2024, 5, 5)),     # Spanish month
    ("12-dez-2023", datetime(2023, 12, 12)),   # Portuguese month
    ("2024-01-31 00:00:00", datetime(2024, 1, 31)),
])
def test_parse_date_formats(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "not a date", "2023-02-30", "31/31/2023", "0"])
def test_bad_dates_return_none(raw):
    assert parse_date(raw) is None


def test_explicit_format_wins():
    assert parse_date("03/04/2023", formats=["%m/%d/%Y"]) == datetime(2023, 3, 4)


# ---------------------------------------------------------------- text
def test_clean_text():
    assert clean_text("  STATE & LOCAL TAXES -  ") == "STATE & LOCAL TAXES"
    assert clean_text("A  B\tC") == "A B C"
    assert clean_text(" ") == ""
