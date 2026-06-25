"""
Unit tests for rag.retriever + rag.vector_store, using a monkeypatched embedder (deterministic,
hand-picked vectors) so these tests exercise the *real* `faiss` library end-to-end without
requiring network access or downloading a real embedding model. Skipped automatically if `faiss`
isn't installed (it is an optional dependency, only required when ENABLE_RAG=True).
"""

import importlib.util
import os
import shutil
import tempfile
import unittest

import numpy as np

from rag.retriever import Document, Retriever

_FAISS_AVAILABLE = importlib.util.find_spec("faiss") is not None


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


# Three orthogonal-ish directions in a tiny 4-dim space, so we can construct queries with a known
# nearest neighbor without needing a real embedding model.
_VEC_CANCELLATION = _unit([1.0, 0.0, 0.0, 0.0])
_VEC_DEPOSIT = _unit([0.0, 1.0, 0.0, 0.0])
_VEC_WEATHER = _unit([0.0, 0.0, 1.0, 0.0])


@unittest.skipUnless(_FAISS_AVAILABLE, "faiss is not installed")
class TestRetrieverWithFakeEmbeddings(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever(embedding_model="bge-small", vector_db="faiss")

        # Monkeypatch the embedder so no network/model download is needed for these tests.
        self._passage_vectors = {
            "Cancel anytime more than 24 hours ahead for a full refund.": _VEC_CANCELLATION,
            "A $300 deposit is required for the premium drone.": _VEC_DEPOSIT,
            "Do not fly in winds over 20mph.": _VEC_WEATHER,
        }
        self.retriever._embedder.encode_passages = lambda texts: np.stack(
            [self._passage_vectors[t] for t in texts]
        )
        self.retriever._embedder.encode_query = self._fake_encode_query

        self.documents = [
            Document(text=t, doc_id=f"doc-{i}") for i, t in enumerate(self._passage_vectors)
        ]
        self.retriever.build_index_from_documents(self.documents)

    def _fake_encode_query(self, query: str) -> np.ndarray:
        if "cancel" in query.lower():
            return _VEC_CANCELLATION
        if "deposit" in query.lower():
            return _VEC_DEPOSIT
        return _VEC_WEATHER

    def test_retrieve_context_returns_expected_shape(self):
        result = self.retriever.retrieve_context("What is your cancellation policy?", top_k=2)
        self.assertEqual(set(result.keys()), {"query", "contexts", "scores"})
        self.assertEqual(result["query"], "What is your cancellation policy?")
        self.assertEqual(len(result["contexts"]), 2)
        self.assertEqual(len(result["scores"]), 2)

    def test_top_result_matches_the_closest_known_vector(self):
        result = self.retriever.retrieve_context("How much is the deposit?", top_k=1)
        self.assertIn("deposit", result["contexts"][0].lower())
        self.assertGreater(result["scores"][0], 0.99)  # exact match in our fake vector space

    def test_score_threshold_filters_out_weak_matches(self):
        result = self.retriever.retrieve_context("How much is the deposit?", top_k=3, score_threshold=0.5)
        # Only the deposit vector is an exact match (score ~1.0); cancellation/weather are
        # orthogonal (score ~0.0) and should be filtered out by the threshold.
        self.assertEqual(len(result["contexts"]), 1)

    def test_metadata_filter(self):
        result = self.retriever._store.search(
            _VEC_DEPOSIT, top_k=3, metadata_filter={"doc_id": "doc-1"}
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].metadata["doc_id"], "doc-1")

    def test_empty_index_returns_empty_result_not_an_error(self):
        empty_retriever = Retriever(embedding_model="bge-small", vector_db="faiss")
        result = empty_retriever.retrieve_context("anything", top_k=5)
        self.assertEqual(result, {"query": "anything", "contexts": [], "scores": []})

    def test_save_and_load_round_trip(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            index_path = os.path.join(tmp_dir, "test_index")
            self.retriever.save_index(index_path)

            reloaded = Retriever(embedding_model="bge-small", vector_db="faiss")
            reloaded.load_index(index_path)
            reloaded._embedder.encode_query = self._fake_encode_query

            result = reloaded.retrieve_context("How much is the deposit?", top_k=1)
            self.assertIn("deposit", result["contexts"][0].lower())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_delete_document_removes_it_from_results(self):
        # doc-1 is the deposit document; after deleting it, deposit queries should not return it.
        self.retriever.delete_documents([1])
        result = self.retriever.retrieve_context("How much is the deposit?", top_k=3, score_threshold=0.5)
        self.assertEqual(result["contexts"], [])


@unittest.skipUnless(_FAISS_AVAILABLE, "faiss is not installed")
class TestUnknownVectorDbBackend(unittest.TestCase):
    def test_chroma_backend_raises_not_implemented(self):
        from rag.vector_store import build_vector_store

        with self.assertRaises(NotImplementedError):
            build_vector_store("chroma", dimension=4)

    def test_unknown_backend_raises_value_error(self):
        from rag.vector_store import build_vector_store

        with self.assertRaises(ValueError):
            build_vector_store("pinecone", dimension=4)


if __name__ == "__main__":
    unittest.main()
