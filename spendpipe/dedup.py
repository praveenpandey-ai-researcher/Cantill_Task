"""Supplier and item de-duplication -> supplier_key, item_key.

SUPPLIERS (config: dedup.supplier)
  1. Normalise names: upper-case, accents/punctuation/legal suffixes removed,
     spaces squashed ('YARF IELD' == 'YARFIELD').
  2. Build a graph: a node per normalised name and per vendor id; a line links
     its name to its vendor id (two names under one id are one supplier).
  3. Fuzzy edges between names: blocking (same first 3 letters) + rapidfuzz
     ratio >= threshold.
  4. Connected components = suppliers.
  5. Optional client rule `unmanaged_bucket`: a supplier whose lines mostly have
     a blank vendor id (no vendor-master record) goes into ONE shared key.
     This reproduces A's golden keys; it is off unless a config turns it on.

ITEMS (config: dedup.item)
  1. Item / material code -> key.
  2. Optional `bucket_rules`: lines matching column conditions get a fixed key
     (A: catalogue/free-text materials without a material type share one key).
  3. DUPLICATE lines inherit the key of the record they duplicate.
  4. No code, but the text matches a coded item's text (TF-IDF cosine) -> that
     item's key.
  5. Still nothing: `no_code` = "blank" (A: the client only keys catalogued
     items) or "text" (default): cluster normalised descriptions (TF-IDF char
     n-grams, cosine >= threshold, numbers must be identical) + components.

Lines matched as DUPLICATE inherit the supplier key of the record they duplicate.
"""
from __future__ import annotations

import logging
import re
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from spendpipe.config import Config
from spendpipe.normalise import norm_supplier, norm_text, squash

log = logging.getLogger(__name__)

SUPPLIER_DEFAULTS = {
    "threshold": 92,             # rapidfuzz ratio on squashed names
    "block_prefix": 3,
    "link_by_id": True,
    "extra_legal_suffixes": [],
    "unmanaged_bucket": None,    # {sources: [..], min_blank_share: 0.5, key: UNMANAGED}
}
ITEM_DEFAULTS = {
    "no_code": "text",           # 'text' | 'blank'
    "text_threshold": 0.85,      # cosine on char n-gram TF-IDF (numbers must match too)
    "text_to_code": {"enabled": True, "threshold": 0.6, "relaxed_threshold": 0.8, "sources": None},
    "no_text_fallback": ["supplier_id", "account"],
    "bucket_rules": [],          # [{key: X, when: {column: value | [values]}, only_with_code: bool}]
}


# --------------------------------------------------------------------------- #
# Graph helper
# --------------------------------------------------------------------------- #
def components(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    """Connected components: node -> component number."""
    idx = {n: i for i, n in enumerate(nodes)}
    if edges:
        a = np.array([idx[x] for x, _ in edges])
        b = np.array([idx[y] for _, y in edges])
    else:
        a = b = np.array([], dtype=int)
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(nodes), len(nodes)))
    _, labels = connected_components(g, directed=False)
    return {n: int(labels[i]) for n, i in idx.items()}


def fuzzy_name_edges(names: list[str], threshold: float, prefix: int) -> list[tuple[str, str]]:
    """Pairs of names with rapidfuzz ratio >= threshold, compared only inside
    blocks of names sharing their first `prefix` characters (keeps it fast)."""
    edges = []
    blocks: dict[str, list[str]] = {}
    for n in names:
        blocks.setdefault(n[:prefix], []).append(n)
    for block in blocks.values():
        if len(block) < 2:
            continue
        sims = process.cdist(block, block, scorer=fuzz.ratio, score_cutoff=threshold)
        ii, jj = np.nonzero(np.triu(sims, k=1))
        edges += [(block[i], block[j]) for i, j in zip(ii, jj)]
    return edges


_NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers(text: str) -> tuple:
    """Numeric signature of an item text: sizes, part numbers, counts.
    'DETONATOR 3.5M 7' -> ('3.5', '7'). Items with different numbers are
    different items, however similar the words are."""
    return tuple(sorted(_NUM.findall(text)))


def _tfidf(texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1,
                           sublinear_tf=True).fit_transform(texts)


def text_clusters(texts: list[str], threshold: float, same_numbers: bool = True) -> dict[str, int]:
    """Cluster texts: TF-IDF on character 2-4-grams, cosine >= threshold,
    connected components. With `same_numbers`, two texts can only join if their
    numeric signatures are identical (stops '3.5M 7' chaining to '3.5M 8')."""
    uniq = sorted(set(t for t in texts if t))
    if not uniq:
        return {}
    if len(uniq) == 1:
        return {uniq[0]: 0}
    X = _tfidf(uniq)
    sig = np.array([hash(numbers(t)) for t in uniq])
    edges_a, edges_b = [], []
    step = 2000
    for start in range(0, len(uniq), step):
        sims = (X[start:start + step] @ X.T).tocoo()
        rows = sims.row + start
        keep = (sims.data >= threshold) & (rows < sims.col)
        if same_numbers:
            keep &= sig[rows] == sig[sims.col]
        edges_a.append(rows[keep])
        edges_b.append(sims.col[keep])
    a = np.concatenate(edges_a)
    b = np.concatenate(edges_b)
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(uniq), len(uniq)))
    _, labels = connected_components(g, directed=False)
    return {t: int(labels[i]) for i, t in enumerate(uniq)}


def map_texts_to_known(query: list[str], known: list[str], threshold: float,
                       relaxed_threshold: float) -> dict[str, str]:
    """For each query text, the most similar known text if it is close enough.

    Accept when cosine >= `threshold` and the numeric signatures agree, or
    cosine >= `relaxed_threshold` regardless of numbers (OCR turns letters into
    digits: 'B1T RCK' for 'BIT RCK'). Returns query -> known text."""
    q = sorted(set(t for t in query if t))
    kn = sorted(set(t for t in known if t))
    if not q or not kn:
        return {}
    exact = set(kn)
    out = {t: t for t in q if t in exact}
    rest = [t for t in q if t not in exact]
    if not rest:
        return out
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(kn + rest)
    K, Q = vec.transform(kn), vec.transform(rest)
    for start in range(0, len(rest), 2000):
        sims = (Q[start:start + 2000] @ K.T).toarray()
        best = sims.argmax(axis=1)
        for i, j in enumerate(best):
            s, t, cand = sims[i, j], rest[start + i], kn[j]
            if s >= relaxed_threshold or (s >= threshold and numbers(t) == numbers(cand)):
                out[t] = cand
    return out


def _majority_name(values: pd.Series) -> str:
    c = Counter(v for v in values if v)
    return c.most_common(1)[0][0] if c else ""


def _inherit(df: pd.DataFrame, keys: pd.Series) -> pd.Series:
    """DUPLICATE lines take the key of the record they duplicate (one level,
    then once more for re-export copies of matched lines)."""
    by_id = pd.Series(keys.values, index=df.record_id.values)
    out = keys.copy()
    for _ in range(2):
        has = df.match_id.isin(by_id.index) & (df.match_id != "")
        out[has] = df.loc[has, "match_id"].map(by_id).values
        by_id = pd.Series(out.values, index=df.record_id.values)
    return out


# --------------------------------------------------------------------------- #
# Suppliers
# --------------------------------------------------------------------------- #
def supplier_keys(df: pd.DataFrame, cfg: Config) -> pd.Series:
    sc = {**SUPPLIER_DEFAULTS, **(cfg.section("dedup").get("supplier") or {})}
    extra = {s.upper() for s in sc["extra_legal_suffixes"]}
    names = df["supplier"].map(lambda s: squash(norm_supplier(s, extra)))
    ids = df["supplier_id"].str.strip().str.upper()

    name_nodes = ["N:" + n for n in names.unique() if n]
    id_nodes = ["I:" + i for i in ids.unique() if i]
    edges = []
    if sc["link_by_id"]:
        pairs = pd.DataFrame({"n": names, "i": ids})
        pairs = pairs[(pairs.n != "") & (pairs.i != "")].drop_duplicates()
        edges += list(zip("N:" + pairs.n, "I:" + pairs.i))
    plain = [n[2:] for n in name_nodes]
    edges += [("N:" + a, "N:" + b) for a, b in
              fuzzy_name_edges(plain, sc["threshold"], sc["block_prefix"])]
    comp = components(name_nodes + id_nodes, edges)

    node = np.where(names != "", "N:" + names, np.where(ids != "", "I:" + ids, ""))
    comp_of_line = pd.Series([comp.get(n, -1) for n in node], index=df.index)

    # readable key: most common raw name in the component, else the vendor id
    label = {}
    tmp = pd.DataFrame({"c": comp_of_line, "name": df["supplier"], "id": ids})
    for c, g in tmp[tmp.c >= 0].groupby("c"):
        nm = _majority_name(g["name"])
        label[c] = norm_supplier(nm, extra).lower() if nm else "id:" + _majority_name(g["id"])
    keys = comp_of_line.map(label).fillna("")

    ub = sc.get("unmanaged_bucket")
    if ub:
        in_src = df.source.isin(ub.get("sources") or df.source.unique())
        share = (ids == "")[in_src].groupby(comp_of_line[in_src]).mean()
        unmanaged = set(share[share > ub.get("min_blank_share", 0.5)].index)
        keys[comp_of_line.isin(unmanaged)] = ub.get("key", "UNMANAGED")
        log.info("supplier unmanaged bucket: %d suppliers, %d lines", len(unmanaged),
                 int(comp_of_line.isin(unmanaged).sum()))

    keys = _inherit(df, keys)
    log.info("suppliers: %d distinct names -> %d supplier keys", names.nunique(), keys.nunique())
    return keys


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #
def _rule_mask(df: pd.DataFrame, when: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in when.items():
        if col not in df:
            log.warning("bucket rule column %s not found; rule ignored", col)
            return pd.Series(False, index=df.index)
        vals = val if isinstance(val, list) else [val]
        mask &= df[col].str.strip().isin([str(v) for v in vals])
    return mask


def item_keys(df: pd.DataFrame, cfg: Config) -> pd.Series:
    ic = {**ITEM_DEFAULTS, **(cfg.section("dedup").get("item") or {})}
    keys = pd.Series("", index=df.index, dtype=object)
    code = df["item_code"].str.strip().str.upper()
    has_code = code != ""
    text = df["description"].map(norm_text)

    # 1. item / material code
    keys[has_code] = "code:" + code[has_code]

    # 2. client bucket rules (config)
    for rule in ic["bucket_rules"]:
        m = _rule_mask(df, rule.get("when", {}))
        if rule.get("only_with_code"):
            m &= has_code
        keys[m] = rule["key"]
        log.info("item bucket rule %s: %d lines", rule["key"], int(m.sum()))

    # 3. duplicates take the key of the record they duplicate, even a blank one
    #    (a tool copy of an ERP service line stays unkeyed, like its original)
    keys = _inherit(df, keys)
    inherited = (df.match_id != "") & df.match_id.isin(df.record_id)

    # 4. no code: text that matches the text of a coded item -> that item's key
    #    (A: tool-only invoices for catalogued materials; 98.9% correct when
    #    tested on tool lines matched to ERP, see DECISIONS.md)
    todo = (keys == "") & (text != "") & ~inherited
    t2c = {**ITEM_DEFAULTS["text_to_code"], **(ic.get("text_to_code") or {})}
    if t2c.get("sources"):          # e.g. only the invoice tool, where codes are not captured
        todo &= df.source.isin(t2c["sources"])
    if t2c.get("enabled", True) and todo.any() and has_code.any():
        known = pd.DataFrame({"t": text[has_code], "k": keys[has_code]})
        text_key = known.groupby("t").k.agg(lambda s: s.value_counts().index[0])
        mp = map_texts_to_known(text[todo].tolist(), list(text_key.index),
                                float(t2c.get("threshold", 0.6)), float(t2c.get("relaxed_threshold", 0.8)))
        got = text[todo].map(mp).map(text_key)
        keys[got.dropna().index] = got.dropna()
        log.info("item text->code: %d of %d code-less lines got a coded item's key",
                 int(got.notna().sum()), int(todo.sum()))

    # 5. still no key: cluster the descriptions (or leave blank, per config)
    todo = (keys == "") & (text != "") & ~inherited
    if ic["no_code"] == "text" and todo.any():
        cl = text_clusters(text[todo].tolist(), float(ic["text_threshold"]))
        tmp = pd.DataFrame({"c": text[todo].map(cl), "t": text[todo]})
        rep = tmp.groupby("c").t.agg(lambda s: s.value_counts().index[0])   # readable key
        keys[todo] = "text:" + tmp.c.map(rep)

    # 6. no code and no text: same supplier + same fallback fields (e.g. GL
    #    account) is most likely the same kind of purchase
    fb = [f for f in ic.get("no_text_fallback", []) if f in df]
    todo = (keys == "") & ~inherited
    if ic["no_code"] == "text" and fb and todo.any():
        parts = df.loc[todo, fb].astype(str).agg("|".join, axis=1)
        keys[todo] = "unknown:" + parts.where(parts.str.strip("|") != "", "")
        keys[keys == "unknown:"] = ""

    log.info("items: %d keys (%d lines without key)", keys[keys != ""].nunique(), int((keys == "").sum()))
    return keys


def dedup(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["supplier_key"] = supplier_keys(df, cfg)
    df["item_key"] = item_keys(df, cfg)
    return df
