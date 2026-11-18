"""The `profile` command guesses columns and drafts a config that runs."""
import yaml

from spendpipe.profile import date_formats, guess_columns, profile
from spendpipe.ingest import read_table
import pandas as pd

LINES = """Line ID;Invoice No;Inv Date;Vendor Name;Vendor No;Item Description;Net Amount;Curr;Status;GL Account
L1;INV-1;31/01/2024;Acme Ltd;V1;Steel bolts M8;1.234,50;EUR;Approved;600100
L2;INV-2;15/02/2024;Beta GmbH;V2;Office cleaning;99,00;EUR;Paid;610200
L3;INV-3;02/03/2024;Gamma SA;V3;Laptop repair;10,00;USD;Cancelled;620300
L4;INV-4;13/03/2024;Acme Ltd;V1;Steel nuts M8;5,00;EUR;On Hold;600100
"""


def test_guess_columns_and_formats(tmp_path):
    f = tmp_path / "lines.csv"
    f.write_text(LINES, encoding="utf-8")
    df = read_table(f)
    m, _ = guess_columns(df)
    assert m["record_id"] == "Line ID"
    assert m["amount"] == "Net Amount"
    assert m["date"] == "Inv Date"
    assert m["supplier"] == "Vendor Name" and m["supplier_id"] == "Vendor No"
    assert m["description"] == "Item Description"
    assert m["currency"] == "Curr" and m["status"] == "Status"
    assert date_formats(df["Inv Date"]) == (["%d/%m/%Y"], True)
    assert date_formats(pd.Series(["01/31/2024", "02/15/2024"])) == (["%m/%d/%Y"], False)


def test_profile_writes_valid_yaml(tmp_path):
    (tmp_path / "lines.csv").write_text(LINES, encoding="utf-8")
    (tmp_path / "tree.csv").write_text("l1,l2\nMaterials,Fasteners\nServices,Cleaning\n", encoding="utf-8")
    out = tmp_path / "c.yaml"
    text = profile(str(tmp_path / "lines.csv"), tree=str(tmp_path / "tree.csv"), out=str(out))
    cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
    src = cfg["sources"][0]
    assert src["columns"]["amount"] == "Net Amount"
    assert src["decimal"] == ","                       # European numbers detected
    assert src["date_formats"] == ["%d/%m/%Y"]
    assert cfg["taxonomy"]["file"] == "tree.csv"
    assert "guessed mapping" in text
