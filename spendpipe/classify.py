"""Classify every included line into the company's tree, with confidence 0-1.

Zero labels needed. The tree is read from a file; no category names in code.

  1. Line text = configured text fields, cleaned, SAP-style abbreviations
     expanded (shared lexicon), e.g. 'MNTN,EQ;F/SERV' -> 'maintenance equipment
     field service'. Blank or vague texts ('Product', 'No object') are marked.
  2. Zero-shot: embed UNIQUE texts and each leaf in two ways (full path
     'L1 > L2 > L3' and leaf name alone) with a multilingual model (default
     BAAI/bge-m3); cosine similarity -> softmax with a temperature set from the
     spread of the similarities -> leaf probabilities, averaged over the two
     leaf views.
  3. (Self-training on pseudo-labels was tried and removed: it reinforced the
     zero-shot's own mistakes and lowered accuracy on both A and B.)
  4. Priors: for each group field (item_key, supplier_key, account, ...) the
     average leaf distribution of its confident lines. Mixed into each line,
     strongly for blank / vague lines, lightly for clear ones.
  5. Optional supervised kNN when labels are given in the config.
  6. Decode top-down: L1 = level-1 node with the highest summed probability,
     then L2 inside it, and so on (best expected accuracy per level).
     confidence = probability of the chosen leaf (sharpened by calibration).

Every step's weights live in the config section `classify`.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spendpipe.config import Config
from spendpipe.embed import Embedder, similarity
from spendpipe.taxonomy import Tree, load_tree

log = logging.getLogger(__name__)

DEFAULTS = {
    "model": "BAAI/bge-m3",
    "backend": "auto",
    "leaf_views": ["path", "leaf"],   # how a leaf is described to the model
    # blend in a character n-gram TF-IDF view (word overlap); 0.2 measured best
    # overall on A and B with labels hidden (also lowers calibration error)
    "tfidf_weight": 0.2,
    "cache_dir": ".cache",
    "text_fields": ["description"],
    "context_fields": [],             # extra text added to the line text (e.g. account_desc)
    "abbreviations": "lexicons/abbreviations.yaml",
    "abbreviations_extra": {},
    "vague_terms": ["product", "products", "parts", "part", "item", "items", "material",
                    "materials", "misc", "miscellaneous", "various", "services", "service",
                    "goods", "no object", "noobject", "n0 object", "other", "others", "general",
                    "sundry", "sundries", "na", "n/a", "none", "test", "-", "."],
    "min_text_chars": 3,
    # Softmax temperature on cosine similarities. 0.01 measured best on A and B;
    # flatter values make top-down decoding favour L1 nodes with many leaves.
    "temperature": 0.01,
    "priors": [                       # group field, weight for clear text, weight for vague text
        # same item -> same category, but only used for blank/vague lines: on
        # clear lines it hurt A (its item buckets mix categories)
        {"field": "item_key", "weight": 0.0, "vague_weight": 2.0},
        # supplier: B L1 spend-weighted 0.879 -> 0.930, neutral on A
        {"field": "supplier_key", "weight": 0.3, "vague_weight": 1.0},
    ],
    "prior_min_lines": 2,
    # Confidence calibration curves: [raw leaf probability, observed accuracy]
    # points, fitted with isotonic regression on A and B pooled
    # (`python analysis/eval_classify.py --fit-calibration`): labels hidden for
    # `zero_shot`, 50% labelled reference for `reference`.
    "calibration": {
        "zero_shot": [[0.05, 0.0], [0.1, 0.0], [0.15, 0.055], [0.35, 0.055], [0.4, 0.101],
                      [0.45, 0.122], [0.5, 0.196], [0.55, 0.251], [0.75, 0.251], [0.8, 0.46],
                      [0.85, 0.685], [0.9, 0.957], [0.95, 0.964], [1.0, 1.0]],
        "reference": [[0.05, 0.0], [0.1, 0.065], [0.15, 0.243], [0.2, 0.271], [0.25, 0.587],
                      [0.3, 0.768], [0.35, 0.864], [0.4, 0.916], [0.45, 0.96], [0.6, 0.961],
                      [0.65, 0.992], [0.8, 0.994], [0.85, 1.0], [1.0, 1.0]],
        "reference_min_similarity": 0.9,
    },
    # Optional supervised kNN layer from labelled examples of the SAME company
    # (A: labels_A, B: seed labels). Loaded through this config from its own
    # data_dir, so --data-dir (blind-test files) does not move it. None = off.
    "reference": None,                # {data_dir, labels, id_column, k, weight, power}
}


# --------------------------------------------------------------------------- #
# Text preparation
# --------------------------------------------------------------------------- #
def load_abbreviations(cfg: Config, cc: dict) -> dict[str, str]:
    abbr: dict[str, str] = {}
    path = cc.get("abbreviations")
    if path:
        full = (cfg.path.parent / path).resolve()
        if full.exists():
            with open(full, encoding="utf-8") as fh:
                abbr = {str(k).upper(): str(v) for k, v in (yaml.safe_load(fh) or {}).items()}
        else:
            log.warning("abbreviation lexicon %s not found", full)
    abbr.update({str(k).upper(): str(v) for k, v in (cc.get("abbreviations_extra") or {}).items()})
    return abbr


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+(?:[.,]\d+)?")


def expand(text: str, abbr: dict[str, str]) -> str:
    """Split SAP short text into tokens and expand known abbreviations.
    'ROD DRIL;SIZE 35 MM' -> 'ROD drill SIZE 35 MM'."""
    s = re.sub(r"P/N\s*:?\s*\S+", " ", text, flags=re.I)        # part numbers carry no meaning
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)                   # CamelCase: VehicleMaintRepair
    # SAP separators split tokens ('74.480,12OZBDY' -> '74.480 12OZBDY'); this
    # measured better than keeping '480,12' as one number (B L1 0.917 vs 0.869)
    s = re.sub(r"[;,/]", " ", s)
    # keep word and number tokens only (punctuation dropped), expand known ones
    return " ".join(abbr.get(tok.upper(), tok) for tok in _TOKEN.findall(s))


def is_vague(text: str, vague: set[str], min_chars: int) -> bool:
    t = re.sub(r"[^a-z/ ]", "", text.lower()).strip()
    if len(re.sub(r"[^a-z]", "", t)) < min_chars:
        return True                                    # blank or numbers only
    return t in vague


# --------------------------------------------------------------------------- #
# Probability helpers
# --------------------------------------------------------------------------- #
def softmax(S: np.ndarray, temperature: float) -> np.ndarray:
    Z = (S - S.max(axis=1, keepdims=True)) / max(temperature, 1e-6)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def calibrate(p: np.ndarray, curve) -> np.ndarray:
    """Piecewise-linear map from raw probability to expected accuracy."""
    pts = np.array(sorted(curve, key=lambda x: x[0]), dtype=float)
    return np.clip(np.interp(np.clip(p, 0, 1), pts[:, 0], pts[:, 1]), 0.0, 1.0)


def group_prior(groups: pd.Series, P: np.ndarray, weight: np.ndarray, min_lines: int) -> np.ndarray:
    """Average distribution of lines in the same group (weighted, e.g. by
    confidence and excluding vague lines). Rows of groups too small -> 0."""
    out = np.zeros_like(P)
    g = groups.fillna("").astype(str).values
    df = pd.DataFrame({"g": g, "w": weight})
    valid = (df.g != "") & (df.w > 0)
    if not valid.any():
        return out
    codes, uniq = pd.factorize(df.g[valid])
    sums = np.zeros((len(uniq), P.shape[1]))
    np.add.at(sums, codes, P[valid.values] * df.w[valid].values[:, None])
    counts = np.bincount(codes, minlength=len(uniq))
    tot = sums.sum(axis=1, keepdims=True)
    dist = np.divide(sums, tot, out=np.zeros_like(sums), where=tot > 0)
    ok = counts >= min_lines
    lookup = {u: i for i, u in enumerate(uniq)}
    for row, key in enumerate(g):
        i = lookup.get(key)
        if i is not None and ok[i]:
            out[row] = dist[i]
    return out


def decode(P: np.ndarray, tree: Tree) -> tuple[list[tuple], np.ndarray]:
    """Top-down: at each level choose the child with the highest summed leaf
    probability inside the chosen parent. Returns paths and leaf confidence."""
    paths = [lf.path for lf in tree.leaves]
    depth = max(len(p) for p in paths)
    n, L = P.shape
    alive = np.ones((n, L), bool)                    # leaves still possible per row
    for d in range(1, depth + 1):
        nodes = sorted({p[:d] for p in paths if len(p) >= d})
        node_ix = {nd: j for j, nd in enumerate(nodes)}
        leaf_node = np.array([node_ix.get(p[:d], -1) if len(p) >= d else -1 for p in paths])
        M = np.zeros((L, len(nodes)))
        M[leaf_node >= 0, leaf_node[leaf_node >= 0]] = 1.0
        scores = (P * alive) @ M                     # summed probability per node
        live_nodes = (alive.astype(float) @ M) > 0
        scores = np.where(live_nodes, scores, -1.0)
        pick = scores.argmax(1)
        has_child = live_nodes.any(1)                # row still has deeper leaves
        keep = leaf_node[None, :] == pick[:, None]
        alive = np.where(has_child[:, None], alive & keep, alive)
    leaf = np.where(alive, P, -1.0).argmax(1)
    return [paths[i] for i in leaf], P[np.arange(n), leaf]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
class Classifier:
    def __init__(self, cfg: Config, tree: Tree | None = None):
        self.cfg = cfg
        cc = cfg.section("classify")
        self.cc = {**DEFAULTS, **cc}
        self.cc["calibration"] = {**DEFAULTS["calibration"], **(cc.get("calibration") or {})}
        self.tree = tree or load_tree(cfg)
        self.abbr = load_abbreviations(cfg, self.cc)
        cache = Path(self.cc["cache_dir"])
        if not cache.is_absolute():
            cache = Path(__file__).resolve().parents[1] / cache
        self.emb = Embedder(self.cc["model"], cache, self.cc["backend"])
        self.info: dict = {"backend": self.emb.backend, "leaves": len(self.tree.leaves)}

    # ---------------------------------------------------------------- texts
    def line_texts(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        fields = [f for f in self.cc["text_fields"] if f in df]
        raw = df[fields].astype(str).agg(" ".join, axis=1).str.strip()
        vague_set = {v.lower() for v in self.cc["vague_terms"]}
        vague = raw.map(lambda t: is_vague(t, vague_set, self.cc["min_text_chars"]))
        text = raw.map(lambda t: expand(t, self.abbr))
        ctx = [f for f in self.cc["context_fields"] if f in df]
        if ctx:
            extra = df[ctx].astype(str).agg(" ".join, axis=1).map(lambda t: expand(t, self.abbr))
            text = (text + " | " + extra).str.strip(" |")
        return text, vague

    # ---------------------------------------------------------------- scoring
    def leaf_view_texts(self, view: str) -> list[str]:
        if view == "leaf":
            return [lf.path[-1] + (f": {lf.description}" if lf.description else "") for lf in self.tree.leaves]
        return [lf.text for lf in self.tree.leaves]         # full path

    def _zero_shot_with(self, emb: Embedder, texts: list[str]) -> np.ndarray:
        views = self.cc["leaf_views"]
        if emb.backend == "tfidf":
            emb.fit_fallback(texts + sum((self.leaf_view_texts(v) for v in views), []))
        Q = emb.encode(texts, "query")
        P = 0
        for view in views:
            # category names are real words: embedded as written (no expansion)
            L = emb.encode(self.leaf_view_texts(view), "query")
            S = similarity(Q, L)
            T = self.cc["temperature"]
            T = max(float(S.std()) / 2, 1e-3) if T == "auto" else float(T)
            P = P + softmax(S, T)
        return P / len(views)

    def zero_shot(self, texts: list[str]) -> np.ndarray:
        """Leaf probabilities for each text, averaged over the leaf views; with
        `tfidf_weight` > 0 blended with a character n-gram TF-IDF view (exact
        word overlap), which is strong where the neural model is weak."""
        P = self._zero_shot_with(self.emb, texts)
        w = float(self.cc.get("tfidf_weight", 0.0))
        if w > 0 and self.emb.backend != "tfidf":
            P = (1 - w) * P + w * self._zero_shot_with(Embedder(backend="tfidf"), texts)
        return P

    # ---------------------------------------------------------------- run
    def classify(self, df: pd.DataFrame,
                 reference: tuple[list[str], np.ndarray] | None = None) -> pd.DataFrame:
        """df: lines to classify (canonical frame). Returns l1..l4, confidence.
        `reference` = (texts, leaf indices) of labelled examples; if None it is
        loaded from the config's `classify.reference` section (if any)."""
        text, vague = self.line_texts(df)
        uniq = sorted(set(text[~vague]))
        log.info("classify: %d lines, %d unique usable texts, %d vague/blank lines",
                 len(df), len(uniq), int(vague.sum()))
        n_leaf = len(self.tree.leaves)
        Pu = self.zero_shot(uniq) if uniq else np.zeros((0, n_leaf))
        index = {t: i for i, t in enumerate(uniq)}

        # line-level text distribution (vague lines: uniform, no information)
        P = np.full((len(df), n_leaf), 1.0 / n_leaf)
        rows = np.array([index.get(t, -1) for t in text])
        has = (~vague.values) & (rows >= 0)
        P[has] = Pu[rows[has]]

        # optional supervised kNN layer from labelled examples (same company)
        if reference is None and self.cc.get("reference"):
            reference = load_reference(self.cfg, self.cc["reference"], self.tree, self)
        knn_best = np.zeros(len(df))
        if reference is not None and len(reference[0]):
            P, has, knn_best = self._knn(text, P, has, reference)
        self.emb.save()

        # group priors (built from clear-text lines, weighted by confidence)
        base_conf = np.where(has, P.max(1), 0.0)
        Pfinal = P.copy()
        for pr in self.cc["priors"]:
            f = pr["field"]
            if f not in df:
                continue
            prior = group_prior(df[f], P, base_conf, int(self.cc["prior_min_lines"]))
            w = np.where(vague.values, float(pr["vague_weight"]), float(pr["weight"]))
            Pfinal = Pfinal + w[:, None] * prior
        Pfinal = Pfinal / Pfinal.sum(axis=1, keepdims=True)

        paths, raw_conf = decode(Pfinal, self.tree)
        # calibrated confidence: raw leaf probability mapped through a curve
        # fitted on A and B with known answers (analysis/eval_classify.py
        # --fit-calibration); a separate curve when a close labelled example
        # exists (kNN layer), because those lines are far more reliable
        cal = self.cc["calibration"]
        near = knn_best >= float(cal.get("reference_min_similarity", 0.9))
        conf = np.where(near, calibrate(raw_conf, cal["reference"]), calibrate(raw_conf, cal["zero_shot"]))
        out = pd.DataFrame(index=df.index)
        out["raw_confidence"] = raw_conf
        out["knn_best"] = knn_best
        depth = max(4, self.tree.levels)
        for d in range(4):
            out[f"l{d + 1}"] = [p[d] if len(p) > d else "" for p in paths]
        out["confidence"] = np.round(conf, 4)
        out["vague_text"] = vague.values
        self.info.update({"unique_texts": len(uniq), "vague_lines": int(vague.sum()),
                          "levels": depth})
        return out

    def _knn(self, text: pd.Series, P: np.ndarray, has: np.ndarray, reference):
        """Mix in a k-nearest-neighbour vote from labelled reference texts.
        Weight per line = weight * (best neighbour similarity) ** power, so an
        identical text dominates and an unfamiliar one barely moves P.
        Vague texts ('No object') are allowed here: the labelled examples show
        what such lines usually are for this company."""
        rc = {"k": 10, "weight": 3.0, "power": 4.0, **(self.cc.get("reference") or {})}
        ref_text, ref_leaf = reference
        n_leaf = P.shape[1]
        # one distribution per unique reference text (labels of identical texts pooled)
        rdf = pd.DataFrame({"t": ref_text, "y": ref_leaf})
        ru = sorted(set(rdf.t))
        rpos = {t: i for i, t in enumerate(ru)}
        Ydist = np.zeros((len(ru), n_leaf))
        np.add.at(Ydist, (rdf.t.map(rpos).values, rdf.y.values.astype(int)), 1.0)
        Ydist /= Ydist.sum(axis=1, keepdims=True)
        R = self.emb.encode(ru, "query")
        qu = sorted(set(text))
        Q = self.emb.encode(qu, "query")
        k = min(int(rc["k"]), len(ru))
        Pk = np.zeros((len(qu), n_leaf))
        best = np.zeros(len(qu))
        for s in range(0, len(qu), 2000):
            S = similarity(Q[s:s + 2000], R)
            top = np.argpartition(-S, k - 1, axis=1)[:, :k]
            sims = np.take_along_axis(S, top, axis=1).clip(min=0)
            wts = sims ** float(rc["power"])
            Pk[s:s + 2000] = np.einsum("nk,nkl->nl", wts, Ydist[top]) / np.maximum(wts.sum(1, keepdims=True), 1e-9)
            best[s:s + 2000] = sims.max(1)
        qi = text.map({t: i for i, t in enumerate(qu)}).values
        w = float(rc["weight"]) * best[qi] ** float(rc["power"])
        P = (P + w[:, None] * Pk[qi]) / (1 + w[:, None])
        log.info("kNN layer: %d labelled texts, median best similarity %.3f", len(ru), float(np.median(best)))
        return P, has | (best[qi] > 0.9), best[qi]


def load_reference(cfg: Config, rc: dict, tree: Tree, clf: "Classifier"):
    """Labelled examples for the kNN layer: the company's own labelled data,
    read through this config but from the reference data_dir (relative to the
    config file), so it is not moved by --data-dir. Returns (texts, leaf idx)."""
    from spendpipe.clean import clean
    from spendpipe.config import load_config
    from spendpipe.ingest import ingest, read_table
    ddir = (cfg.path.parent / rc["data_dir"]).resolve()
    lab_file = ddir / rc["labels"]
    if not lab_file.exists():
        log.warning("reference labels %s not found; kNN layer off", lab_file)
        return None
    try:
        rcfg = load_config(cfg.path, data_dir=ddir)
        rdf = clean(ingest(rcfg), rcfg)
    except Exception as exc:                       # noqa: BLE001 - never crash on extras
        log.warning("reference data could not be loaded (%s); kNN layer off", exc)
        return None
    lab = read_table(lab_file)
    lab = lab.rename(columns={rc.get("id_column", "record_id"): "record_id"})
    if "include" in lab:
        lab = lab[lab["include"].astype(str) == "1"]
    cols = [c for c in lab.columns if re.match(r"(category_)?l(evel)?_?\d", c.lower())]
    lab["path"] = lab[cols].apply(lambda r: tuple(str(v).strip() for v in r if str(v).strip()), axis=1)
    leaf_ix = {lf.path: i for i, lf in enumerate(tree.leaves)}
    lab["y"] = lab["path"].map(leaf_ix)
    m = rdf.merge(lab[["record_id", "y"]], on="record_id")
    missing = int(m.y.isna().sum())
    m = m[m.y.notna()]
    if missing:
        log.warning("reference: %d labelled lines have a path not in the tree; ignored", missing)
    text, _ = clf.line_texts(m)
    log.info("reference: %d labelled lines from %s", len(m), lab_file.name)
    return text.tolist(), m.y.astype(int).values


def classify(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Classify included lines; excluded lines get blank categories."""
    df = df.copy()
    for c in ("l1", "l2", "l3", "l4"):
        df[c] = ""
    df["confidence"] = 0.0
    inc = df["include"] == 1
    if inc.any():
        clf = Classifier(cfg)
        res = clf.classify(df[inc])
        for c in ("l1", "l2", "l3", "l4", "confidence"):
            df.loc[inc, c] = res[c]
        df.loc[inc, "vague_text"] = res["vague_text"]
        return df, clf.info
    return df, {}
