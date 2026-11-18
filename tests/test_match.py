"""Count-once matching on a tiny synthetic two-system company."""
from spendpipe.config import load_config
from spendpipe.pipeline import run

ERP = """line_id,po,item,value,vendor,text,qty,date
E1,4510000001,10.0,1000.00,ACME (PTY) LTD,BOLT ROOF 1.8M,10,2023-01-10
E2,4510000002,10.0,500.00,ACME (PTY) LTD,RESIN CAPSULE,5,
E3,4510000003,10.0,200.00,BETA CC,GLOVES,2,
E4,4510000003,20.0,200.00,BETA CC,GLOVES,2,
E5,4510000009,10.0,300.00,GAMMA,MESH,1,
"""
# T1 PO+line+amount (gross incl 15% VAT) | T2 PO formatted, no line
# T3 + T4 two identical lines, each must claim its own ERP receipt (E3, E4)
# T5 no PO -> supplier+amount fallback to E5 | T6 PO not in ERP -> tool-only
# T7 re-export of T6 with other formats -> DUPLICATE | T8 cancelled
TOOL = """id,inv,date,supplier,po,text,qty,gross,vat,status
T1,INV/2023/1001,12/01/2023,Acme,4510000001/10.0,BOLT ROOF 1.8M,10,"1,150.00",15%,Paid
T2,IN0001002,,ACME PTY LTD,PO#4510000002,RESIN CAPSULE,5,575.00,,Approved
T3,1003,,Beta,4510000003,GLOVES,2,230.00,0.15,Paid
T4,1003,,Beta,4510000003,GLOVES,2,230.00,0.15,Paid
T5,1004,,GAMMA,,MESH,1,345.00,15,Paid
T6,SI-2023-1005,,Delta,4519999999,PUMP,1,1150.00,15%,Paid
T7,TAX INV 1005,,DELTA,PO 4519999999,PUMP,1,1150.00,15,Paid
T8,1006,,Acme,4510000001,BOLT ROOF 1.8M,10,1150.00,15%,Cancelled
"""
CFG = """
company: T
reporting_currency: ZAR
data_dir: .
taxonomy: {file: tree.csv}
sources:
  - name: ERP
    file: erp.csv
    role: primary
    id_prefix: "ERP:"
    columns: {record_id: line_id, amount: value, po_number: po, po_line: item,
              supplier: vendor, description: text, quantity: qty, date: date}
    currency: {fixed: ZAR}
  - name: TOOL
    file: tool.csv
    role: secondary
    id_prefix: "TOOL:"
    columns: {record_id: id, amount: gross, invoice_number: inv, date: date, supplier: supplier,
              po_number: po, description: text, quantity: qty, vat_rate: vat, status: status}
    date_formats: ["%d/%m/%Y"]
    amount: {gross: true, default_vat_rate: 0.15}
    currency: {fixed: ZAR}
matching:
  enabled: true
  po_pattern: "(\\\\d{10})"
"""


def test_count_once(tmp_path):
    (tmp_path / "erp.csv").write_text(ERP)
    (tmp_path / "tool.csv").write_text(TOOL)
    (tmp_path / "tree.csv").write_text("l1\nX\n")
    (tmp_path / "c.yaml").write_text(CFG)
    df, rep = run(load_config(tmp_path / "c.yaml"))
    r = df.set_index("record_id")

    assert r.loc["TOOL:T1", "match_id"] == "ERP:E1"
    assert r.loc["TOOL:T1", "match_pass"] == "po_line_amount"
    assert r.loc["TOOL:T2", "match_id"] == "ERP:E2"
    assert {r.loc["TOOL:T3", "match_id"], r.loc["TOOL:T4", "match_id"]} == {"ERP:E3", "ERP:E4"}
    assert r.loc["TOOL:T5", "match_id"] == "ERP:E5"
    assert r.loc["TOOL:T5", "match_pass"] == "supplier_amount"
    assert r.loc["TOOL:T6", "dup_reason"] == ""                 # tool-only spend
    assert r.loc["TOOL:T7", "match_pass"] == "reexported_invoice"
    assert r.loc["TOOL:T8", "status_reason"] == "CANCELLED"
    assert r.loc["TOOL:T8", "dup_reason"] == ""                 # cancelled lines never claim
    assert rep["matching"]["secondary_only_spend"] == 1000.0     # T6 net of VAT
