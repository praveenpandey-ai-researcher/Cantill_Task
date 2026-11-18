"""PO / invoice / supplier / text normalisation."""
import pytest

from spendpipe.normalise import norm_invoice, norm_po, norm_po_line, norm_supplier, norm_text


@pytest.mark.parametrize("raw", ["4510008616", "PO#4510008616", "PO 4510008616",
                                 "4510008616/10.0", "po-4510008616 ", "0004510008616"])
def test_po_formats_normalise_to_same_key(raw):
    assert norm_po(raw) == "4510008616"


def test_po_blank_and_pattern():
    assert norm_po("") == "" and norm_po("n/a") == ""
    assert norm_po("PO#4510008616/10.0", pattern=r"(\d{10})") == "4510008616"


@pytest.mark.parametrize("raw,expected", [("10.0", "10"), ("4510008616/20.0", "20"),
                                          ("010", "10"), ("", "")])
def test_po_line(raw, expected):
    assert norm_po_line(raw) == expected


@pytest.mark.parametrize("raw", ["INV/2023/759857", "INV/2024/759857", "SI-2022-759857",
                                 "IN0759857", "TAX INV 759857", "759857"])
def test_invoice_formats_normalise_to_same_key(raw):
    assert norm_invoice(raw) == "759857"


@pytest.mark.parametrize("a,b", [
    ("OAKENT MINING SA (PTY) LTD", "Oakent Mining"),
    ("KINORA  INDUSTRIAL SUPPLIES ENGINEERING", "Kinora Industrial Supplies Engineering"),
    ("Wexvex CC", "WEXVEX"),
    ("Siemens AG", "SIEMENS"),
    ("Grupo Bimbo S.A. de C.V.", "GRUPO BIMBO"),
])
def test_supplier_legal_suffixes(a, b):
    assert norm_supplier(a) == norm_supplier(b)


def test_supplier_keeps_single_word_name():
    assert norm_supplier("Limited") == "LIMITED"     # never strip to nothing


def test_text_decimal_comma():
    assert norm_text("DETONATOR, EXPLOSIVES;3,5M,7") == norm_text("DETONATOR, EXPLOSIVES;3.5M,7")
