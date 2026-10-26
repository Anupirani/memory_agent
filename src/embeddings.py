import hashlib
import os
import re

import numpy as np

STOPWORDS = set("""a an the is are was were am be been i me my mine we our you your it its
of to in on at for with and or but not no do does did have has had this that these those
what which who whom how when where why can could should would will just so as by from about
user""".split())


class HashEmbedder:
    """Dependency-free embedder: signed hashing of words, word pairs and character trigrams.
    Good for testing and for keyword-style overlap; it does NOT understand synonyms."""
    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
        feats = list(words) + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [padded[i:i + 3] for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f in self._features(text):
                h = int(hashlib.md5(f.encode("utf-8")).hexdigest(), 16)
                out[row, h % self.dim] += 1.0 if (h >> 64) % 2 == 0 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class FastEmbedder:
    """Real semantic embeddings, run locally with ONNX (no API key, no GPU)."""
    name = "fastembed"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding  # pip install fastembed
        self._model = TextEmbedding(model_name=model)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def get_embedder():
    """EMBEDDER=hash (default) or EMBEDDER=fastembed, set in .env."""
    if os.getenv("EMBEDDER", "hash").lower() == "fastembed":
        return FastEmbedder()
    return HashEmbedder()