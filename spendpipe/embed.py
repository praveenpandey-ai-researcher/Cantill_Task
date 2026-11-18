"""Text embeddings with a disk cache and an offline fallback.

Backend 1: sentence-transformers multilingual model (default
           intfloat/multilingual-e5-small). Downloaded once into the cache.
Backend 2: TF-IDF on character n-grams, used automatically when the model
           cannot be loaded (no internet, blocked DLLs...). Works offline.

Only UNIQUE texts are embedded, and vectors are cached on disk per model, so
re-runs (and the blind test on A2/B2) only embed texts never seen before.
All vectors are L2-normalised: cosine similarity = dot product.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model: str = "intfloat/multilingual-e5-small", cache_dir: str | Path = ".cache",
                 backend: str = "auto", batch_size: int = 64):
        self.model_name = model
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.backend = None
        self._st = None
        self._tfidf = None
        self.is_e5 = "e5" in model.lower()
        if backend in ("auto", "sentence-transformers"):
            try:
                os.environ.setdefault("HF_HOME", str(self.cache_dir / "hf"))
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(model, cache_folder=str(self.cache_dir / "hf"))
                self.backend = "sentence-transformers"
            except Exception as exc:          # noqa: BLE001 - any failure -> fallback
                if backend == "sentence-transformers":
                    raise
                log.warning("embedding model %s unavailable (%s); using TF-IDF fallback",
                            model, type(exc).__name__)
        if self._st is None:
            self.backend = "tfidf"
        slug = (model if self.backend != "tfidf" else "tfidf").replace("/", "__")
        self._cache_file = self.cache_dir / "embeddings" / f"{slug}.npz"
        self._cache: dict[str, np.ndarray] = {}
        if self.backend != "tfidf":
            self._load_cache()
        log.info("embedder backend: %s", self.backend)

    # ----------------------------------------------------------------- cache
    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if self._cache_file.exists():
            try:
                z = np.load(self._cache_file, allow_pickle=False)
                self._cache = dict(zip(z["keys"].tolist(), z["vecs"]))
                log.info("embedding cache: %d vectors", len(self._cache))
            except Exception as exc:          # noqa: BLE001 - corrupt cache -> rebuild
                log.warning("embedding cache unreadable (%s); rebuilding", exc)

    def save(self) -> None:
        if self.backend == "tfidf" or not self._cache:
            return
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._cache.keys()))
        vecs = np.stack(list(self._cache.values())).astype(np.float32)
        np.savez(self._cache_file, keys=keys, vecs=vecs)

    # ----------------------------------------------------------------- encode
    def fit_fallback(self, corpus: list[str]) -> None:
        """TF-IDF needs a vocabulary: fit on all texts that will be compared."""
        if self.backend != "tfidf":
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                      lowercase=True).fit(corpus)

    def encode(self, texts: list[str], kind: str = "query") -> np.ndarray:
        """kind = 'query' (line text) or 'passage' (category text); e5 models
        were trained with these prefixes."""
        if self.backend == "tfidf":
            if self._tfidf is None:
                self.fit_fallback(texts)
            X = self._tfidf.transform(texts)
            return X                     # sparse, already L2-normalised
        prefix = f"{kind}: " if self.is_e5 else ""
        full = [prefix + t for t in texts]
        keys = [self._key(self.model_name + "|" + f) for f in full]
        missing = sorted({(k, f) for k, f in zip(keys, full) if k not in self._cache})
        if missing:
            log.info("embedding %d new texts (%d cached)", len(missing), len(full) - len(missing))
            vecs = self._st.encode([f for _, f in missing], batch_size=self.batch_size,
                                   normalize_embeddings=True, show_progress_bar=False,
                                   convert_to_numpy=True)
            for (k, _), v in zip(missing, vecs):
                self._cache[k] = v.astype(np.float32)
        return np.stack([self._cache[k] for k in keys]) if keys else np.zeros((0, 1), np.float32)


def similarity(a, b) -> np.ndarray:
    """Cosine similarity matrix for dense or sparse L2-normalised rows."""
    s = a @ b.T
    return s.toarray() if hasattr(s, "toarray") else np.asarray(s)
