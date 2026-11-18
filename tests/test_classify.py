"""Taxonomy loading, text preparation, decoding, calibration and an offline
end-to-end classification (TF-IDF backend: no model download needed)."""
import numpy as np
import pandas as pd

from spendpipe.classify import Classifier, calibrate, decode, expand, is_vague
from spendpipe.config import load_config
from spendpipe.taxonomy import Leaf, Tree, load_tree, path_text

TREE = """level_1,level_2,level_3,level_4,description
Food,Produce,Fruit,Citrus,oranges lemons limes
Food,Produce,Fruit,Berries,
Food,Produce,Vegetables,,carrots onions
Food,Dairy,,,milk cheese butter
Services,Cleaning,Cleaning,,office cleaning janitorial
Services,Security,,,guards cctv
"""


def make_cfg(tmp_path, extra=""):
    (tmp_path / "tree.csv").write_text(TREE)
    (tmp_path / "x.csv").write_text("id,amt\n1,1\n")
    (tmp_path / "c.yaml").write_text(f"""
taxonomy: {{file: tree.csv}}
sources: [{{name: S, file: x.csv, columns: {{record_id: id, amount: amt}}}}]
classify:
  backend: tfidf
  abbreviations: null
  cache_dir: {tmp_path.as_posix()}/cache
{extra}
""")
    return load_config(tmp_path / "c.yaml")


def test_tree_ragged_four_levels(tmp_path):
    tree = load_tree(make_cfg(tmp_path))
    paths = {lf.path for lf in tree.leaves}
    assert tree.levels == 4
    assert ("Food", "Produce", "Fruit", "Citrus") in paths      # 4 levels
    assert ("Food", "Dairy") in paths                           # branch ends at L2
    assert ("Food", "Produce", "Fruit") not in paths            # inner node, not a leaf
    citrus = next(lf for lf in tree.leaves if lf.path[-1] == "Citrus")
    assert "oranges" in citrus.text                             # description column used


def test_path_text_drops_repeats():
    assert path_text(("Services", "Cleaning", "Cleaning")) == "Services > Cleaning"


def test_expand_and_vague():
    abbr = {"MNTN": "maintenance", "EQ": "equipment", "REPR": "repair"}
    assert expand("MNTN,EQ;F/SERV,OT", abbr) == "maintenance equipment F SERV OT"
    assert expand("ROD DRIL;SIZE 35 MM", {"DRIL": "drill"}) == "ROD drill SIZE 35 MM"
    assert "P/N" not in expand("REPR;VALVE;P/N:JHF14852600,BANK", abbr)
    assert expand("VehicleMaintRepairSV", {}) == "Vehicle Maint Repair SV"
    vague = {"product", "no object"}
    assert is_vague("Product", vague, 3) and is_vague("No Object", vague, 3)
    assert is_vague("0054492532", vague, 3) and is_vague("", vague, 3)
    assert not is_vague("GLOVES LEATHER", vague, 3)


def test_decode_top_down_sums_children():
    tree = Tree(levels=2, leaves=[Leaf(("A", "a1"), ""), Leaf(("A", "a2"), ""), Leaf(("B", "b1"), "")])
    # B's single leaf is the most likely leaf, but A as a whole is more likely
    P = np.array([[0.3, 0.3, 0.4]])
    paths, conf = decode(P, tree)
    assert paths[0][0] == "A" and conf[0] == 0.3


def test_calibrate_interpolates_and_clips():
    curve = [[0.0, 0.0], [0.5, 0.2], [1.0, 1.0]]
    assert np.allclose(calibrate(np.array([0.25, 0.75, 1.5]), curve), [0.1, 0.6, 1.0])


def test_classify_offline_end_to_end(tmp_path):
    cfg = make_cfg(tmp_path)
    df = pd.DataFrame({
        "description": ["oranges and lemons", "fresh carrots", "office cleaning service",
                        "security guards night shift", "Product", "cheese and milk"],
        "supplier_key": ["s1", "s2", "s3", "s4", "s1", "s5"],
        "item_key": ["", "", "", "", "", ""],
    })
    res = Classifier(cfg).classify(df)
    got = res[["l1", "l2", "l3", "l4"]].apply(lambda r: tuple(v for v in r if v), axis=1).tolist()
    assert got[0] == ("Food", "Produce", "Fruit", "Citrus")
    assert got[2][0] == "Services" and got[3] == ("Services", "Security")
    assert got[4][0] == "Food"                  # vague 'Product' follows its supplier (s1: citrus)
    assert res.confidence.between(0, 1).all()
    assert (res.l4[[2, 3]] == "").all()         # blank below the branch's last level
