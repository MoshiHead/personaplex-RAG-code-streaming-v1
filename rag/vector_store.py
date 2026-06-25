"""
FAISS-backed vector store (Phase 3 of the project plan).

Chroma is configured as a recognized `RAGConfig.vector_db` value but is **not implemented yet** --
`build_vector_store("chroma", ...)` raises a clear `NotImplementedError` rather than silently
falling back to something else. FAISS was prioritized per the project brief and is implemented
first, fully.

Storage model: one FAISS `IndexIDMap(IndexFlatIP)` (inner product over L2-normalized vectors is
exactly cosine similarity) for the vectors, plus a parallel dict of metadata keyed by the same
integer ids. FAISS itself has no concept of metadata, so a JSON sidecar file is always saved/loaded
alongside the binary `.faiss` index file.

`import rag.vector_store` does not import `faiss` at module scope -- `faiss` is only imported
inside `FaissVectorStore.__init__`/`.load()`, so this file can be imported safely even on a machine
without `faiss-cpu` installed, as long as no FAISS store is actually constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Optional, Sequence

import numpy as np


@dataclass
class SearchResult:
    id: int
    score: float
    text: str
    metadata: dict


class FaissVectorStore:
    """Create / save / load / update / delete, per Phase 3's requirements."""

    def __init__(self, dimension: int):
        import faiss  # lazy import -- see module docstring

        self._faiss = faiss
        self.dimension = dimension
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        self._metadata: dict[int, dict] = {}
        self._next_id = 0

    # ---- create / add --------------------------------------------------------------------
    def add(
        self,
        vectors: np.ndarray,
        texts: Sequence[str],
        metadatas: Optional[Sequence[dict]] = None,
    ) -> list[int]:
        """Adds `vectors` (N, dim) with accompanying `texts`/`metadatas`. Returns the assigned ids
        (monotonically increasing, never reused even after `delete()`, so stale references to a
        deleted id fail loudly via a missing-metadata lookup rather than silently pointing at
        unrelated new content)."""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        vectors = np.asarray(vectors, dtype=np.float32)
        assert len(vectors) == len(texts) == len(metadatas), "vectors/texts/metadatas length mismatch"

        ids = np.arange(self._next_id, self._next_id + len(vectors), dtype=np.int64)
        if len(ids) > 0:
            self.index.add_with_ids(vectors, ids)
        for doc_id, text, meta in zip(ids, texts, metadatas):
            self._metadata[int(doc_id)] = {"text": text, **meta}
        self._next_id += len(vectors)
        return ids.tolist()

    # ---- search -----------------------------------------------------------------------------
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Cosine-similarity top-k search (assumes `query_vector` and the stored vectors are both
        L2-normalized -- `rag.embeddings` guarantees this). `metadata_filter` is a simple equality
        filter (`{"topic": "policy"}` keeps only results whose metadata has `topic == "policy"`);
        FAISS has no native filtering, so we over-fetch candidates and filter in Python."""
        if self.index.ntotal == 0:
            return []

        fetch_k = min(top_k * 5 if metadata_filter else top_k, self.index.ntotal)
        scores, ids = self.index.search(query_vector[None, :].astype(np.float32), fetch_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            meta = self._metadata.get(int(idx), {})
            if metadata_filter and not _matches_filter(meta, metadata_filter):
                continue
            if score_threshold is not None and score < score_threshold:
                continue
            results.append(
                SearchResult(id=int(idx), score=float(score), text=meta.get("text", ""), metadata=meta)
            )
            if len(results) >= top_k:
                break
        return results

    # ---- update / delete ---------------------------------------------------------------------
    def update(self, doc_id: int, vector: np.ndarray, text: str, metadata: Optional[dict] = None) -> None:
        """FAISS flat indexes have no in-place update; implemented as delete + re-add at the same id."""
        self.delete([doc_id])
        self.index.add_with_ids(
            np.asarray(vector, dtype=np.float32)[None, :], np.array([doc_id], dtype=np.int64)
        )
        self._metadata[doc_id] = {"text": text, **(metadata or {})}

    def delete(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        self.index.remove_ids(np.array(ids, dtype=np.int64))
        for doc_id in ids:
            self._metadata.pop(int(doc_id), None)

    # ---- persistence ---------------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Writes `<path>.faiss` (binary index) and `<path>.meta.json` (metadata + bookkeeping)."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._faiss.write_index(self.index, path + ".faiss")
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {"dimension": self.dimension, "next_id": self._next_id, "metadata": self._metadata},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "FaissVectorStore":
        import faiss

        with open(path + ".meta.json", encoding="utf-8") as f:
            payload = json.load(f)
        store = cls(dimension=payload["dimension"])
        store.index = faiss.read_index(path + ".faiss")
        store._metadata = {int(k): v for k, v in payload["metadata"].items()}
        store._next_id = payload["next_id"]
        return store


def _matches_filter(metadata: dict, filt: dict) -> bool:
    return all(metadata.get(k) == v for k, v in filt.items())


def build_vector_store(backend: str, dimension: int) -> FaissVectorStore:
    """Factory matching `RAGConfig.vector_db`."""
    if backend == "faiss":
        return FaissVectorStore(dimension)
    if backend == "chroma":
        raise NotImplementedError(
            "VECTOR_DB='chroma' is not implemented yet -- only 'faiss' is available in this "
            "increment. See docs/ARCHITECTURE_REPORT.md, Phase 3."
        )
    raise ValueError(f"Unknown vector_db backend '{backend}'.")
