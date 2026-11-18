"""Supplier and item de-duplication on small synthetic data."""
import pandas as pd

from spendpipe.dedup import (components, fuzzy_name_edges, item_keys, map_texts_to_known,
                             numbers, supplier_keys, text_clusters)


class FakeCfg:
    def __init__(self, dedup):
        self._d = {"dedup": dedup}

    def section(self, name):
        return self._d.get(name) or {}


def frame(**cols):
    n = len(next(iter(cols.values())))
    base = {"record_id": [f"R{i}" for i in range(n)], "source": ["S"] * n, "match_id": [""] * n,
            "supplier": [""] * n, "supplier_id": [""] * n, "item_code": [""] * n,
            "description": [""] * n, "account": [""] * n}
    base.update(cols)
    return pd.DataFrame(base)


def test_components_and_fuzzy_edges():
    names = ["YARFIELD", "YARFIELDS", "YARVEX"]
    comp = components(names, fuzzy_name_edges(names, 90, 3))
    assert comp["YARFIELD"] == comp["YARFIELDS"] != comp["YARVEX"]


def test_supplier_spellings_and_ids_merge():
    df = frame(supplier=["Acme (Pty) Ltd", "ACME", "A C M E", "Beta CC", "Beta Holdings", ""],
               supplier_id=["", "", "", "V1", "V1", "V9"])
    k = supplier_keys(df, FakeCfg({"supplier": {}}))
    assert k[0] == k[1] == k[2]          # suffixes and spacing
    assert k[3] == k[4]                  # different names, same vendor id
    assert k[5] == "id:V9"               # id only (company B style)
    assert k[0] != k[3]


def test_unmanaged_bucket():
    df = frame(supplier=["Acme", "Acme", "Beta", "Gamma"], supplier_id=["", "", "V1", ""])
    cfg = FakeCfg({"supplier": {"unmanaged_bucket": {"min_blank_share": 0.5, "key": "UNM"}}})
    k = supplier_keys(df, cfg)
    assert k.tolist()[0] == k.tolist()[1] == k.tolist()[3] == "UNM"
    assert k[2] != "UNM"


def test_numbers_signature_blocks_size_variants():
    assert numbers("DETONATOR 3.5M 7") != numbers("DETONATOR 3.5M 8")
    cl = text_clusters(["DETONATOR EXPLOSIVES 3.5M 7", "DETONATOR EXPLOSIVE 3.5M 7",
                        "DETONATOR EXPLOSIVES 3.5M 8"], 0.8)
    assert cl["DETONATOR EXPLOSIVES 3.5M 7"] == cl["DETONATOR EXPLOSIVE 3.5M 7"]
    assert cl["DETONATOR EXPLOSIVES 3.5M 7"] != cl["DETONATOR EXPLOSIVES 3.5M 8"]


def test_map_texts_handles_ocr():
    # OCR swapped I -> 1 (seen in A's tool export); unrelated text is not mapped
    known = ["REPR CYLINDER DRILL FEED STL CAST", "ROD DRIL", "RESIN CAPSULE"]
    mp = map_texts_to_known(["REPR CYL1NDER DRILL FEED STL CAST", "HAND TOOLS"], known, 0.6, 0.8)
    assert mp == {"REPR CYL1NDER DRILL FEED STL CAST": "REPR CYLINDER DRILL FEED STL CAST"}


def test_item_keys_code_then_text_then_fallback():
    df = frame(item_code=["M1", "M1", "", "", "", ""],
               description=["BOLT ROOF 1.8M", "bolt roof 1,8m", "BOLT ROOF 1.8M",
                            "GLOVES LEATHER", "", ""],
               supplier_id=["V1"] * 4 + ["V2", "V2"], account=["", "", "", "", "GL1", "GL1"])
    k = item_keys(df, FakeCfg({"item": {}}))
    assert k[0] == k[1] == k[2] == "code:M1"     # text of a code-less line maps to the code
    assert k[3].startswith("text:")
    assert k[4] == k[5] == "unknown:V2|GL1"


def test_item_bucket_rule_and_blank_mode():
    df = frame(item_code=["M1", "M2", ""], description=["A", "B", "C"])
    df["x_type"] = ["", "", ""]
    cfg = FakeCfg({"item": {"no_code": "blank", "text_to_code": {"enabled": False},
                            "bucket_rules": [{"key": "BKT", "only_with_code": True,
                                              "when": {"x_type": ""}}]}})
    assert item_keys(df, cfg).tolist() == ["BKT", "BKT", ""]


def test_duplicates_inherit_keys():
    df = frame(supplier=["Acme", "Totally Different"], item_code=["M1", ""],
               description=["X", "Y"])
    df.loc[1, "match_id"] = "R0"
    assert supplier_keys(df, FakeCfg({})).nunique() == 1
    assert item_keys(df, FakeCfg({})).tolist() == ["code:M1", "code:M1"]
