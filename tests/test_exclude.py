"""Status rules, NON_PROCUREMENT lexicon and exclusion precedence."""
from pathlib import Path

import pandas as pd
import pytest

from spendpipe.config import load_config
from spendpipe.exclude import DEFAULT_STATUS, NonProcurementRules, combine_reasons, status_reason

LEXICON = (Path(__file__).resolve().parents[1] / "configs" / "lexicons" / "non_procurement.yaml")


# ---------------------------------------------------------------- status
@pytest.fixture
def rules():
    return {k: [v.lower() for v in vals] for k, vals in DEFAULT_STATUS.items()}


@pytest.mark.parametrize("value,expected", [
    ("Paid", ""), ("APPROVED", ""), ("Cancelled", "CANCELLED"), ("canceled", "CANCELLED"),
    ("On Hold", "NOT_APPROVED"), ("on_hold", "NOT_APPROVED"), ("Rejected", "NOT_APPROVED"),
    ("Disputed", "NOT_APPROVED"), ("Pagado", ""), ("Anulado", "CANCELLED"),
    ("Something new", "NOT_APPROVED"),   # unknown -> only approved/paid count
    ("", ""),                            # no status in the system -> counts
])
def test_status(rules, value, expected):
    assert status_reason(value, rules) == expected


# ---------------------------------------------------------------- lexicon
@pytest.fixture(scope="module")
def np_rules(tmp_path_factory):
    d = tmp_path_factory.mktemp("cfg")
    (d / "x.csv").write_text("id,amt\n1,1\n")
    (d / "t.csv").write_text("l1\nA\n")
    (d / "c.yaml").write_text(f"""
taxonomy: {{file: t.csv}}
sources: [{{name: S, file: x.csv, columns: {{record_id: id, amount: amt}}}}]
exclusions:
  non_procurement:
    lexicon: '{LEXICON.as_posix()}'
    account_desc_patterns: ['^IC\\d', '^I/C\\b']
""")
    return NonProcurementRules(load_config(d / "c.yaml"))


@pytest.mark.parametrize("text,group", [
    ("STATE & LOCAL TAXES", "tax"),
    ("Sales Tax Expense", "tax"),
    ("GST INPUT TAX", "tax"),
    ("ACCRUED SALES & USE TAX", "tax"),
    ("2024 - KDV %20", "tax"),
    ("Impuestos municipales", "tax"),
    ("Umsatzsteuer Q1", "tax"),
    ("Intercompany recharge Q2", "intercompany"),
    ("Bank charges March", "finance"),
    ("Interest on loan", "finance"),
    ("Royalties 2024", "royalty"),
    ("Regalías mineras", "royalty"),
    ("Partner billing Q3", "revenue_partner"),
    ("Monthly payroll", "payroll_people"),
])
def test_lexicon_hits(np_rules, text, group):
    assert np_rules.text_rule(text) == group


@pytest.mark.parametrize("text", [
    "TAX SERVICES AND GARNISHMENTS",     # bought service
    "BLANKET PO FOR TAX WORK",
    "ASSISTANCE WITH PROPERTY TAX V",
    "PAYTECH TAX CONSULTANT",
    "GLOBALVIEW PAYROLL SERVICES",
    "CUSTOMS CLEARENCE GRAPHICS 2024 VAT",
    "IMPORT CUSTOMS DUTY COSTS",         # customs duty is a category in B's tree
    "HEAVY DUTY SCREWDRIVER",
    "ABRASIVE DISC VERY FINE GRIT",
    "BANKERS BOX R-KIVE",
    "BANCO ESCALERA DE ALUMINIO",        # Spanish: bench
    "CISCO ISR 1100 ROUTER",
    "ROD ANCHR;DIA:20MM,1800MM",
    "",
])
def test_lexicon_does_not_hit(np_rules, text):
    assert np_rules.text_rule(text) == ""


def test_account_desc_rule(np_rules):
    df = pd.DataFrame({"account": ["", "", ""], "account_desc": ["IC06", "I/C Payable", "AP - Other"],
                       "supplier_id": "", "supplier": "", "description": ["x", "y", "STATE & LOCAL TAXES"]})
    assert np_rules.apply(df).tolist() == ["account_desc", "account_desc", "text:tax"]


# ---------------------------------------------------------------- precedence
def test_precedence():
    df = pd.DataFrame({"status_reason": ["CANCELLED", "", "", ""],
                       "dup_reason": ["DUPLICATE", "DUPLICATE", "", ""],
                       "np_reason": ["NON_PROCUREMENT", "NON_PROCUREMENT", "NON_PROCUREMENT", ""]})
    # a duplicated tax line is NON_PROCUREMENT: its original is not counted either
    assert combine_reasons(df, list(df.columns)).tolist() == [
        "CANCELLED", "NON_PROCUREMENT", "NON_PROCUREMENT", ""]
