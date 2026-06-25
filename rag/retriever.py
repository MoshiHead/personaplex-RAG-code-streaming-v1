"""
Retrieval layer (Phase 5 of the project plan): ties `rag.embeddings` + `rag.vector_store` together
behind one call, `Retriever.retrieve_context(...)`, returning the exact shape requested in the
project brief:

    {"query": "...", "contexts": [...], "scores": [...]}

Also provides document ingestion (`Retriever.build_index_from_documents`) and index persistence,
since Phase 5 retrieval is useless without Phase 4's "document ingestion" half being wired to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .embeddings import EmbeddingModel
from .vector_store import FaissVectorStore, build_vector_store


@dataclass
class Document:
    """One ingestable unit of knowledge. `doc_id` is caller-assigned (e.g. a stable slug from a
    knowledge-base JSON file) and stored in metadata for traceability; the vector store's own
    integer id (assigned at insertion time) is what `retrieve_context` results key off of."""

    text: str
    doc_id: str
    metadata: dict | None = None


class Retriever:
    """Bundles one embedding model + one vector store. Construct via `Retriever(config)`, then
    either `build_index_from_documents(...)` (fresh ingest) or `load_index(path)` (reuse a
    previously built+saved index) before calling `retrieve_context(...)`."""

    def __init__(self, embedding_model: str = "bge-small", vector_db: str = "faiss"):
        self.embedding_model_name = embedding_model
        self.vector_db_backend = vector_db
        self._embedder = EmbeddingModel(embedding_model)
        self._store: FaissVectorStore | None = None

    @property
    def is_ready(self) -> bool:
        return self._store is not None and self._store.index.ntotal > 0

    def build_index_from_documents(self, documents: Sequence[Document]) -> int:
        """Embeds every document's text and adds it to a freshly created vector store. Returns the
        number of documents indexed. This is the "document ingestion" + "create index" half of
        Phase 3/4 combined into one call."""
        texts = [d.text for d in documents]
        metadatas = [{"doc_id": d.doc_id, **(d.metadata or {})} for d in documents]

        vectors = self._embedder.encode_passages(texts)
        self._store = build_vector_store(self.vector_db_backend, dimension=vectors.shape[1])
        self._store.add(vectors, texts, metadatas)
        return len(documents)

    def save_index(self, path: str) -> None:
        if self._store is None:
            raise RuntimeError("No index to save -- call build_index_from_documents() or load_index() first.")
        self._store.save(path)

    def load_index(self, path: str) -> None:
        self._store = FaissVectorStore.load(path)

    def delete_documents(self, ids: Sequence[int]) -> None:
        if self._store is None:
            raise RuntimeError("No index loaded.")
        self._store.delete(ids)

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[dict] = None,
    ) -> dict:
        """Phase 5 entry point. Returns:
            {"query": query, "contexts": [text, ...], "scores": [float, ...]}
        with `contexts`/`scores` empty (not an error) if no index is loaded or nothing matches --
        callers (e.g. Mode B/C policy code) should treat an empty result as "nothing relevant
        found" and fall back to the bare persona prompt, not crash.
        """
        if self._store is None or self._store.index.ntotal == 0:
            return {"query": query, "contexts": [], "scores": []}

        query_vector = self._embedder.encode_query(query)
        results = self._store.search(
            query_vector, top_k=top_k, score_threshold=score_threshold, metadata_filter=metadata_filter
        )
        return {
            "query": query,
            "contexts": [r.text for r in results],
            "scores": [r.score for r in results],
        }
