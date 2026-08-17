"""Dense + sparse embedding generation via BGE-M3.

BGE-M3 produces both a dense vector and a lexical (sparse/BM25-style)
representation from a single forward pass, which is why one provider
class exposes both `embed_dense` and `embed_sparse`.

The heavy `FlagEmbedding` import (pulls in torch/transformers) is deferred
to `BGEM3EmbeddingProvider.__init__` so modules importing
`EmbeddingProvider` for typing stay lightweight, and tests can use
`FakeEmbeddingProvider` without real model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from loguru import logger

from app.retrieval.device import get_device

# Dimensionality of BAAI/bge-m3's dense output; used when creating the
# Qdrant collection's dense vector config.
BGE_M3_DENSE_DIM = 1024


@dataclass(frozen=True)
class SparseVector:
    """Qdrant's sparse vector wire format: parallel index/value arrays."""

    indices: list[int]
    values: list[float]


class EmbeddingProvider(Protocol):
    """Interface every embedding backend implements, so retrieval code
    never depends on a concrete model implementation.
    """

    def embed_dense(self, texts: list[str]) -> list[list[float]]: ...

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]: ...


class BGEM3EmbeddingProvider:
    """Production embedding provider backed by `BAAI/bge-m3` via
    `FlagEmbedding`.

    Requires network access to HuggingFace Hub on first load (weights are
    then cached locally). Device handling is automatic: CUDA when a usable
    GPU is present, otherwise CPU.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool | None = None) -> None:
        try:
            from FlagEmbedding import (
                BGEM3FlagModel,
            )
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required for BGEM3EmbeddingProvider. "
                "Install it with `pip install FlagEmbedding`."
            ) from exc

        device = get_device()
        # FP16 mixed precision is only beneficial/supported on CUDA. If the
        # caller didn't explicitly opt in/out, enable it when a GPU exists.
        if use_fp16 is None:
            use_fp16 = device == "cuda"
        elif use_fp16 and device != "cuda":
            logger.warning(
                "use_fp16=True requested but no CUDA device is available; "
                "falling back to FP32 on CPU."
            )
            use_fp16 = False

        logger.info(
            "Loading BGE-M3 embedding model ({}) on {} (fp16={})...",
            model_name,
            device,
            use_fp16,
        )
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=[device])
        logger.info("BGE-M3 model loaded on {}", device)

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        output = self._model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return [vec.tolist() for vec in output["dense_vecs"]]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        output = self._model.encode(texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        sparse_vectors: list[SparseVector] = []
        for lexical_weights in output["lexical_weights"]:
            # BGE-M3 returns {token_id (as str): weight}; Qdrant wants
            # parallel int-index/float-value arrays.
            indices = [int(token_id) for token_id in lexical_weights.keys()]
            values = [float(weight) for weight in lexical_weights.values()]
            sparse_vectors.append(SparseVector(indices=indices, values=values))
        return sparse_vectors


class FakeEmbeddingProvider:
    """Deterministic, dependency-free embedding provider for tests.

    Hashes tokens into fixed-size vectors: not semantically meaningful,
    but deterministic, so retrieval/Qdrant plumbing can be tested without
    real model weights.
    """

    def __init__(self, dense_dim: int = BGE_M3_DENSE_DIM) -> None:
        self._dense_dim = dense_dim

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vector(text) for text in texts]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [self._hash_sparse(text) for text in texts]

    def _hash_vector(self, text: str) -> list[float]:
        import hashlib

        vector = [0.0] * self._dense_dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dense_dim
            vector[index] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def _hash_sparse(self, text: str) -> SparseVector:
        import hashlib
        from collections import Counter

        counts: Counter[int] = Counter()
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % 250_002  # approximate BGE-M3 vocab size
            counts[index] += 1
        indices = list(counts.keys())
        values = [float(v) for v in counts.values()]
        return SparseVector(indices=indices, values=values)
