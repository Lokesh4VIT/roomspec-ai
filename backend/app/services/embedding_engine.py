"""Multimodal embeddings.

`ClipEmbedder` puts room photos and catalog text in the same 512-d CLIP space.
`HashEmbedder` is a dependency-free fallback (signed feature hashing of text
n-grams) so the service, CI and tests run without downloading model weights.
Both return L2-normalized float32 vectors.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.core.config import get_settings

log = logging.getLogger(__name__)


def _l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norm, 1e-12)).astype(np.float32)


class Embedder(Protocol):
    name: str
    dim: int
    supports_images: bool

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    def embed_image(self, image_rgb: np.ndarray) -> np.ndarray: ...


class HashEmbedder:
    name = "hash"
    supports_images = False

    _TOKEN = re.compile(r"[a-z0-9]+")
    _STOP = frozenset(["a", "an", "the", "and", "or", "with", "for", "of", "on", "in", "to", "cm", "x", "by", "is"])

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        tokens = [t for t in self._TOKEN.findall(text.lower()) if t not in self._STOP]
        feats = list(tokens)
        feats += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for tok in tokens:  # char trigrams tolerate plurals / typos ("walnuts", "grey"/"gray")
            padded = f"#{tok}#"
            feats += [f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                weight = 0.35 if feat.startswith("c:") else 1.0
                out[row, h % self.dim] += weight if (h >> 63) & 1 else -weight
        return _l2(out)

    def embed_image(self, image_rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Hash embedder has no image tower; use colour-derived text instead")


class ClipEmbedder:
    name = "clip"
    supports_images = True

    def __init__(self, model_name: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = int(self.model.config.projection_dim)
        self._lock = threading.Lock()

    @staticmethod
    def _as_tensor(out):
        # transformers >= 5 may return a model output object instead of a tensor.
        return out if hasattr(out, "cpu") else out.pooler_output

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        with self._lock, self._torch.inference_mode():
            feats = self._as_tensor(self.model.get_text_features(**inputs))
        return _l2(feats.cpu().numpy())

    def embed_image(self, image_rgb: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        with self._lock, self._torch.inference_mode():
            feats = self._as_tensor(self.model.get_image_features(**inputs))
        return _l2(feats.cpu().numpy())[0]


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_backend in ("auto", "clip"):
        try:
            emb = ClipEmbedder(s.clip_model_name)
            log.info("Loaded CLIP embedder %s (dim=%s)", s.clip_model_name, emb.dim)
            return emb
        except Exception as exc:  # torch missing, no network for weights, etc.
            if s.embedding_backend == "clip":
                raise
            log.warning("CLIP unavailable (%s); falling back to hash embedder", exc)
    return HashEmbedder(s.embedding_dim)


_LEX_TOKEN = re.compile(r"[a-z]+")
_LEX_STOP = frozenset(
    {"a", "an", "the", "and", "or", "with", "for", "of", "on", "in", "to", "by", "is", "are", "as", "at"}
    | {"from", "this", "that", "it", "its", "unit", "units", "kitchen", "photo"}
)


def _stem(token: str) -> str:
    # Tiny plural folding ("cabinets" -> "cabinet", "cupboards" -> "cupboard"); enough for catalog text.
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def lexical_vector(text: str) -> tuple[list[int], list[float]]:
    """Sparse term-frequency vector for Qdrant's IDF-weighted lexical index (BM25-style).

    Dense CLIP embeddings handle paraphrases ("espresso timber" -> walnut) but blur exact
    catalog vocabulary ("shaker", "navy", "sage"); the sparse index covers that gap.
    """
    counts: dict[int, float] = {}
    for tok in _LEX_TOKEN.findall(text.lower()):
        if tok in _LEX_STOP or len(tok) < 2:
            continue
        idx = int.from_bytes(hashlib.blake2b(_stem(tok).encode(), digest_size=4).digest(), "little")
        counts[idx] = counts.get(idx, 0.0) + 1.0
    indices = sorted(counts)
    # Sub-linear TF saturation, in the spirit of BM25's k1 term.
    return indices, [round(1.0 + np.log(counts[i]), 4) for i in indices]


def lexical_document(row: dict) -> str:
    fields = (
        row["part_name"],
        row["finish_style"],
        row["category"],
        row.get("material", ""),
        row.get("description", ""),
    )
    return " ".join(fields)


def module_document(row: dict) -> str:
    """Caption-style text for a catalog module.

    CLIP was trained on image captions, so "a photo of ..." phrasing retrieves better than
    spec-sheet text (P@5 0.90 vs 0.76 on the finish benchmark). Kept under CLIP's 77 tokens.
    """
    blurb = row.get("description", "")
    for marker in ("Finish: ", "Surface: "):
        if marker in blurb:
            blurb = blurb.split(marker, 1)[1]
    finish, category = row["finish_style"].lower(), row["category"].lower()
    return f"a photo of a {finish} kitchen {category}, {blurb} {row.get('material', '')}"
