"""Classification accuracy with labels HIDDEN (zero-shot), scored afterwards.

  A: classified from its tree alone, scored against labels_A (ERP lines).
  B: classified from its tree alone, scored against seed_labels (included lines)
     as a stand-in for the unlabelled company C.

Reports accuracy per level (line-weighted and spend-weighted, as the blind test
weights by spend) and a calibration table (confidence bin vs accuracy).

    python analysis/eval_classify.py --company A B
    python analysis/eval_classify.py --company B --set classify.temperature=0.03
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from spendpipe.classify import Classifier
from spendpipe.config import load_config
from spendpipe.pipeline import run

_PREP: dict = {}


def prepared(company: str):
    """Pipeline up to de-dup (cached in memory), plus gold paths."""
    if company in _PREP:
        return _PREP[company]
    cfg = load_config(f"configs/company_{company}.yaml")
    df, _ = run(cfg, stop_before="classify")
    lc = cfg.section("labels")
    lab = pd.read_csv(cfg.file(lc["file"]), dtype=str, keep_default_na=False)
    lab = lab.rename(columns={lc.get("id_column", "record_id"): "record_id"})
    if "include" in lab:
        lab = lab[lab.include == "1"]
    cols = [c for c in lab.columns if c.startswith("category_l")]
    lab["gold"] = lab[cols].apply(lambda r: tuple(v for v in r if v), axis=1)
    _PREP[company] = (cfg, df, lab[["record_id", "gold"]])
    return _PREP[company]


def set_path(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def evaluate(company: str, overrides: list[str], quiet: bool = False,
             reference_split: float = 0.0) -> dict:
    """reference_split = 0: labels fully hidden (zero-shot, the C scenario).
    reference_split = f: a random share f of the labelled lines is the kNN
    reference and only the other lines are scored (the A2 / B2 scenario)."""
    cfg, df, gold = prepared(company)
    cfg = copy.copy(cfg)
    cfg.raw = copy.deepcopy(cfg.raw)
    set_path(cfg.raw, "classify.reference", None)          # never read labels from config here
    for ov in overrides:
        k, v = ov.split("=", 1)
        set_path(cfg.raw, k, yaml.safe_load(v))
    inc = df[df.include == 1]
    t0 = time.time()
    clf = Classifier(cfg)
    reference = None
    if reference_split > 0:
        ref_ids = gold.record_id.sample(frac=reference_split, random_state=0)
        ref = inc[inc.record_id.isin(ref_ids)].merge(gold, on="record_id")
        leaf_ix = {lf.path: i for i, lf in enumerate(clf.tree.leaves)}
        ref = ref[ref.gold.isin(leaf_ix)]
        reference = (clf.line_texts(ref)[0].tolist(), ref.gold.map(leaf_ix).values)
        gold = gold[~gold.record_id.isin(ref_ids)]
    res = clf.classify(inc, reference=reference)
    secs = time.time() - t0
    out = inc[["record_id", "spend_raw"]].join(res)
    m = out.merge(gold, on="record_id")
    evaluate.last = m          # kept for --fit-calibration
    pred = m[["l1", "l2", "l3", "l4"]].apply(lambda r: tuple(v for v in r if v), axis=1)
    w = m.spend_raw.abs().fillna(0)
    rep = {"company": company, "mode": f"reference {reference_split:.0%}" if reference_split else "labels hidden",
           "lines": len(m), "seconds": round(secs, 1), "backend": clf.emb.backend}
    for d in (1, 2, 3, 4):
        if not any(len(g) >= d for g in m.gold):
            continue
        ok = np.array([p[:d] == g[:d] for p, g in zip(pred, m.gold)])
        rep[f"L{d}"] = round(ok.mean(), 3)
        rep[f"L{d}_spend"] = round(float(np.average(ok, weights=w)) if w.sum() else ok.mean(), 3)
    leaf_ok = np.array([p == g for p, g in zip(pred, m.gold)])
    m["leaf_ok"] = leaf_ok
    bins = pd.cut(m.confidence, [0, .2, .4, .6, .8, 1.0], include_lowest=True)
    cal = pd.DataFrame({"bin": bins, "ok": leaf_ok}).groupby("bin", observed=True).ok.agg(["size", "mean"])
    rep["mean_conf"] = round(float(m.confidence.mean()), 3)
    rep["leaf_acc"] = round(float(leaf_ok.mean()), 3)
    rep["ece"] = round(ece(m.confidence.values, leaf_ok), 3)
    if not quiet:
        print(json.dumps(rep))
        print("calibration (confidence bin -> lines, leaf accuracy):")
        print(cal.round(3).to_string())
    return rep


def ece(conf: np.ndarray, ok: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: average |confidence - accuracy| over bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            err += sel.mean() * abs(conf[sel].mean() - ok[sel].mean())
    return float(err)


def fit_curve(x: np.ndarray, y: np.ndarray) -> list[list[float]]:
    """Isotonic regression of correctness on raw confidence, sampled on a
    fixed 0.05 grid (only where data exists; ends clamp)."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x, y)
    xs = np.round(np.arange(0, 1.0001, 0.05), 2)
    xs = xs[(xs >= x.min() - 0.05) & (xs <= x.max() + 0.05)]
    return [[float(a), round(float(b), 3)] for a, b in zip(xs, iso.predict(xs))]


def fit_calibration(overrides: list[str]) -> None:
    """Fit both calibration curves on A and B pooled (each company weighted
    equally) and print them as YAML for the classifier defaults."""
    ident = ["classify.calibration.zero_shot=[[0,0],[1,1]]", "classify.calibration.reference=[[0,0],[1,1]]"]
    curves = {}
    for mode, split in (("zero_shot", 0.0), ("reference", 0.5)):
        xs, ys, ws = [], [], []
        for c in ("A", "B"):
            evaluate(c, overrides + ident, quiet=True, reference_split=split)
            m = evaluate.last
            if mode == "reference":
                m = m[m.knn_best >= 0.9]
            xs.append(m.raw_confidence.values)
            ys.append(m.leaf_ok.values.astype(float))
            ws.append(np.full(len(m), 1.0 / max(len(m), 1)))
        x, y = np.concatenate(xs), np.concatenate(ys)
        curves[mode] = fit_curve(x, y)
        print(f"{mode}: ECE raw {ece(x, y.astype(bool)):.3f}")
    print(yaml.safe_dump({"calibration": curves}, default_flow_style=None, sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--company", nargs="+", default=["A", "B"])
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. classify.temperature=0.03")
    ap.add_argument("--reference-split", type=float, default=0.0,
                    help="share of labelled lines used as kNN reference (0 = labels hidden)")
    ap.add_argument("--fit-calibration", action="store_true",
                    help="fit confidence calibration curves on A and B and print them")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.fit_calibration:
        fit_calibration(args.set)
        return
    for c in args.company:
        evaluate(c, args.set, reference_split=args.reference_split)


if __name__ == "__main__":
    main()
