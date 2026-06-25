"""
Embedding backend for the retrieval layer (Phase 4 of the project plan).

Wraps `sentence-transformers`, which natively loads BGE and E5 checkpoints in addition to generic
"sentence-transformers/*" models -- so "sentence-transformers support", "BGE support", and "E5
support" are satisfied by one implementation plus a short model-id + prefix registry, rather than
three parallel code paths.

Import of `sentence_transformers` is lazy (inside `_load_model`), so `import rag.embeddings` costs
nothing and requires nothing extra when `ENABLE_RAG=False` -- consistent with the rest of this
package.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

# Short config names (as used by RAGConfig.embedding_model) -> (HF model id, query prefix, passage
# prefix). BGE and E5 both document specific prefixing conventions for best retrieval quality:
#   - BGE: prefix the *query* only with an instruction; passages are encoded as-is.
#   - E5: prefix *both* queries and passages, with "query: " / "passage: " respectively.
# Getting these wrong doesn't raise an error -- it just silently degrades retrieval quality -- so
# we encode the correct convention here once instead of leaving it to every caller to remember.
_MODEL_REGISTRY: dict[str, tuple[str, str, str]] = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "bge-base": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "bge-large": ("BAAI/bge-large-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5-small": ("intfloat/e5-small-v2", "query: ", "passage: "),
    "e5-base": ("intfloat/e5-base-v2", "query: ", "passage: "),
    "e5-large": ("intfloat/e5-large-v2", "query: ", "passage: "),
    "sentence-transformers": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
}


def resolve_model_id(name: str) -> tuple[str, str, str]:
    """Map a short `RAGConfig.embedding_model` name to (hf_model_id, query_prefix, passage_prefix).

    Names not found in the registry are treated as a literal HF model id with no prefixing -- this
    lets advanced users point at any sentence-transformers-compatible model without us needing to
    special-case every possible id.
    """
    if name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[name]
    return name, "", ""


@lru_cache(maxsize=4)
def _load_model(model_id: str):
    # Imported lazily: importing rag.embeddings should never require sentence-transformers (and
    # therefore torch) to be installed unless an embedding call is actually made.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


@dataclass
class EmbeddingModel:
    """Thin, cached wrapper around one sentence-transformers checkpoint + its prefix convention."""

    name: str = "bge-small"

    def __post_init__(self):
        self.model_id, self._query_prefix, self._passage_prefix = resolve_model_id(self.name)

    @property
    def dimension(self) -> int:
        return _load_model(self.model_id).get_sentence_embedding_dimension()

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of documents/passages for indexing. Returns float32 (N, dim), L2-normalized
        so a FAISS inner-product index over these vectors computes cosine similarity directly."""
        model = _load_model(self.model_id)
        prefixed = [f"{self._passage_prefix}{t}" for t in texts]
        embeddings = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Embed a single query string for retrieval. Returns float32 (dim,), L2-normalized."""
        model = _load_model(self.model_id)
        prefixed = f"{self._query_prefix}{text}"
        embedding = model.encode([prefixed], normalize_embeddings=True, convert_to_numpy=True)[0]
        return np.asarray(embedding, dtype=np.float32)


def build_embeddings(texts: Sequence[str], model_name: str = "bge-small") -> np.ndarray:
    """Phase 4 entry point: embed a batch of passages (documents) for indexing."""
    return EmbeddingModel(model_name).encode_passages(list(texts))


def query_embeddings(query: str, model_name: str = "bge-small") -> np.ndarray:
    """Phase 4 entry point: embed a single query string for retrieval."""
    return EmbeddingModel(model_name).encode_query(query)
