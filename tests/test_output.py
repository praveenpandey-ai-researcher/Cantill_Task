"""The output contract: columns, one row per record, spend 0 when excluded."""
import pandas as pd
import pytest

from spendpipe.output import COLUMNS, build_output, check_output


def frame():
    return pd.DataFrame({
        "record_id": ["ERP:1", "TOOL:2", "TOOL:3", "4"],
        "include": [1, 0, 0, 0],
        "exclusion_reason": ["", "DUPLICATE", "CANCELLED", "NON_PROCUREMENT"],
        "match_pass": ["", "po_amount", "", ""],
        "np_rule": ["", "", "", "account_desc"],
        "supplier_key": ["acme", "acme", "beta", "gamma"],
        "item_key": ["code:M1", "code:M1", "", ""],
        "spend_raw": [100.0, 100.0, 50.0, float("nan")],
        "l1": ["A", "A", "", ""], "l2": ["a1", "a1", "", ""], "l3": ["", "", "", ""], "l4": ["", "", "", ""],
        "confidence": [0.8, 0.8, 0.0, 0.0],
    })


def test_build_output_contract():
    out = build_output(frame())
    assert list(out.columns) == COLUMNS
    assert out.spend.tolist() == [100.0, 0.0, 0.0, 0.0]                 # excluded -> 0
    assert out.loc[1, "l1"] == ""                                        # no category when excluded
    assert out.confidence.tolist() == [0.8, 0.97, 1.0, 0.97]             # exclusion confidence by rule
    check_output(out, n_input=4)


def test_check_output_catches_missing_rows_and_duplicate_ids():
    out = build_output(frame())
    with pytest.raises(RuntimeError, match="output rows"):
        check_output(out, n_input=5)
    out.loc[1, "record_id"] = "ERP:1"
    with pytest.raises(RuntimeError, match="duplicated"):
        check_output(out, n_input=4)
