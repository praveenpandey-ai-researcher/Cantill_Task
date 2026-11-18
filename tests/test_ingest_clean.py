"""End-to-end ingest + clean on a tiny synthetic company: must never crash."""
import math

from spendpipe.clean import clean
from spendpipe.config import load_config
from spendpipe.ingest import ingest


def test_messy_file_is_handled(tmp_path):
    # semicolon file, an unknown extra column, a missing mapped column (vat),
    # bad date, bad amount, blank currency, missing FX month
    (tmp_path / "lines.csv").write_text(
        "id;amt;dt;ccy;rate;txt;weird_extra\n"
        "1;1.150,00;31/01/2024;EUR;;Schrauben M8;x\n"
        "2;abc;2024-13-45;USD;17.3;;y\n"
        "3;100;;;;  bolts - ;z\n",
        encoding="utf-8")
    (tmp_path / "fx.csv").write_text("currency,month,units_per_usd\nEUR,2023-06,0.9\n", encoding="utf-8")
    (tmp_path / "tree.csv").write_text("l1\nA\n", encoding="utf-8")
    (tmp_path / "c.yaml").write_text("""
company: T
data_dir: .
reporting_currency: USD
taxonomy: {file: tree.csv}
fx: {table_file: fx.csv}
sources:
  - name: S
    file: lines.csv
    columns: {record_id: id, amount: amt, date: dt, currency: ccy, fx_rate: rate,
              description: txt, vat_rate: no_such_column}
    amount: {gross: true, default_vat_rate: 0.15}
""", encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")
    df = clean(ingest(cfg), cfg).set_index("record_id")

    # row 1: European number, VAT 15% removed, EUR via nearest table month
    assert math.isclose(df.loc["1", "net_amount"], 1000.0)
    assert df.loc["1", "fx_source"].startswith("table_nearest")
    assert math.isclose(df.loc["1", "spend_raw"], 1000.0 / 0.9)
    # row 2: unparsable amount and date are flagged, not fatal
    assert "AMOUNT_UNPARSABLE" in df.loc["2", "dq_flags"]
    assert "DATE_UNPARSABLE" in df.loc["2", "dq_flags"]
    # row 3: blank currency -> reporting currency; text cleaned
    assert df.loc["3", "fx_source"] == "same_ccy"
    assert df.loc["3", "description"] == "bolts"
    # unmapped column kept for later stages
    assert "x_weird_extra" in df.columns
