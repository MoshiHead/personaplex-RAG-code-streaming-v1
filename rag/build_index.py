"""
Ingestion CLI for Phase 3/4 ("document ingestion", "create index", "save index"): reads a
knowledge-base JSON file (a list of `{"doc_id", "topic", "text"}` objects -- see
`rag/data/aero_rentals_kb.json` for the sample knowledge base used to validate Mode C) and writes a
FAISS index + metadata sidecar to disk via `rag.retriever.Retriever`.

Usage:
    python -m rag.build_index \
        --kb rag/data/aero_rentals_kb.json \
        --out rag_indexes/aero_rentals \
        --embedding-model bge-small \
        --vector-db faiss
"""

from __future__ import annotations

import argparse
import json
import time

from .retriever import Document, Retriever


def load_documents(kb_path: str) -> list[Document]:
    with open(kb_path, encoding="utf-8") as f:
        raw = json.load(f)
    documents = []
    for entry in raw:
        metadata = {k: v for k, v in entry.items() if k not in ("doc_id", "text")}
        documents.append(Document(text=entry["text"], doc_id=entry["doc_id"], metadata=metadata))
    return documents


def build_index(kb_path: str, out_path: str, embedding_model: str = "bge-small", vector_db: str = "faiss") -> dict:
    """Builds and saves an index from `kb_path`. Returns a small report dict (also what the CLI
    prints), useful for notebook cells that want to assert on it (e.g. "index has N documents")."""
    documents = load_documents(kb_path)

    t0 = time.monotonic()
    retriever = Retriever(embedding_model=embedding_model, vector_db=vector_db)
    n_indexed = retriever.build_index_from_documents(documents)
    build_time_s = time.monotonic() - t0

    retriever.save_index(out_path)

    return {
        "kb_path": kb_path,
        "out_path": out_path,
        "embedding_model": embedding_model,
        "vector_db": vector_db,
        "documents_indexed": n_indexed,
        "build_time_s": build_time_s,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", required=True, help="Path to a knowledge-base JSON file.")
    parser.add_argument("--out", required=True, help="Output path prefix for the saved index.")
    parser.add_argument("--embedding-model", default="bge-small")
    parser.add_argument("--vector-db", default="faiss")
    args = parser.parse_args()

    report = build_index(args.kb, args.out, args.embedding_model, args.vector_db)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
