"""
Unit tests for rag.server_integration.RAGSession, using the same plain-Python `FakeLMGen`/
`FakeTokenizer` pattern as rag/tests/test_injection_manager.py, plus a hand-rolled fake retriever
(no faiss/sentence-transformers needed) so these tests run anywhere, independent of optional RAG
dependencies being installed.
"""

import shutil
import tempfile
import unittest

from rag.config import InjectionMode, RAGConfig
from rag.injection_manager import InjectionRequest
from rag.server_integration import RAGSession


class FakeLMGen:
    def __init__(self):
        self.calls = []
        self.reset_streaming_called = False
        self._streaming_state = None  # no real RingKVCache -- inspect_kv_cache() should degrade

    def step(self, input_tokens=None, moshi_tokens=None, text_token=None, return_embeddings=False):
        self.calls.append({"text_token": text_token})

    def reset_streaming(self):
        self.reset_streaming_called = True


class FakeTokenizer:
    def encode(self, text: str) -> list:
        return [ord(ch) for ch in text]


class FakeRetriever:
    """Stands in for rag.retriever.Retriever without needing faiss/sentence-transformers."""

    def __init__(self, canned_result: dict):
        self._canned_result = canned_result
        self.queries_seen = []

    def retrieve_context(self, query, top_k=5, score_threshold=None, metadata_filter=None):
        self.queries_seen.append(query)
        return self._canned_result


def _make_session(config: RAGConfig, log_dir: str, retriever=None) -> tuple:
    lm_gen = FakeLMGen()
    session = RAGSession(
        config=config,
        lm_gen=lm_gen,
        text_tokenizer=FakeTokenizer(),
        make_zero_audio_frame=lambda: "ZERO",
        make_silence_audio_frame=lambda: "SINE",
    )
    if retriever is not None:
        session.retriever = retriever
    return session, lm_gen


class TestRAGSessionModeC(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = RAGConfig(
            enable_rag=True,
            injection_mode=InjectionMode.PERSONA_RAG,
            log_dir=self.tmp_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_disabled_rag_skips_injection_and_logs_skip_reason(self):
        config = RAGConfig(enable_rag=False, log_dir=self.tmp_dir)
        session, lm_gen = _make_session(config, self.tmp_dir, retriever=FakeRetriever({"query": "q", "contexts": [], "scores": []}))

        result = session.inject_persona_compatible_knowledge("What is the deposit?")

        self.assertIn("skipped", result["injection_strategy"])
        self.assertEqual(len(lm_gen.calls), 0)
        self.assertFalse(lm_gen.reset_streaming_called)

    def test_no_retriever_loaded_skips_injection(self):
        session, lm_gen = _make_session(self.config, self.tmp_dir, retriever=None)
        result = session.inject_persona_compatible_knowledge("What is the deposit?")
        self.assertIn("skipped", result["injection_strategy"])
        self.assertEqual(len(lm_gen.calls), 0)

    def test_empty_retrieval_result_skips_injection(self):
        retriever = FakeRetriever({"query": "q", "contexts": [], "scores": []})
        session, lm_gen = _make_session(self.config, self.tmp_dir, retriever=retriever)
        result = session.inject_persona_compatible_knowledge("What is the deposit?")
        self.assertIn("skipped (no contexts", result["injection_strategy"])
        self.assertEqual(len(lm_gen.calls), 0)

    def test_successful_retrieval_forces_tokens_through_lm_gen(self):
        retriever = FakeRetriever(
            {"query": "q", "contexts": ["A $300 deposit is required."], "scores": [0.92]}
        )
        session, lm_gen = _make_session(self.config, self.tmp_dir, retriever=retriever)

        result = session.inject_persona_compatible_knowledge("How much is the deposit?")

        self.assertEqual(result["injection_strategy"], "persona_rag (blocking burst, same <system> mechanism as persona prompt)")
        self.assertGreater(len(lm_gen.calls), 0)
        self.assertFalse(lm_gen.reset_streaming_called)  # never resets the live cache
        self.assertEqual(result["injected_token_count"], len(lm_gen.calls))
        self.assertEqual(result["retrieved_contexts"], ["A $300 deposit is required."])
        self.assertEqual(result["retrieved_scores"], [0.92])
        self.assertIsNotNone(result["retrieval_latency_s"])
        self.assertIsNotNone(result["injection_latency_s"])
        # Real LMGen attrs aren't present on FakeLMGen -> kv_cache_status must degrade, not raise.
        self.assertFalse(result["kv_cache_status"]["available"])

    def test_log_record_is_persisted_to_disk(self):
        retriever = FakeRetriever({"query": "q", "contexts": ["fact one"], "scores": [0.8]})
        session, _ = _make_session(self.config, self.tmp_dir, retriever=retriever)
        session.inject_persona_compatible_knowledge("a question")

        rows = session.logger.read_all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["user_query"], "a question")


class TestRAGSessionIncrementalQueue(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = RAGConfig(enable_rag=True, injection_mode=InjectionMode.DYNAMIC_RUNTIME, log_dir=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_consume_one_tick_is_a_safe_noop_with_nothing_queued(self):
        session, lm_gen = _make_session(self.config, self.tmp_dir)
        self.assertFalse(session.consume_one_tick())
        self.assertEqual(len(lm_gen.calls), 0)

    def test_queued_job_drains_one_token_per_tick(self):
        session, lm_gen = _make_session(self.config, self.tmp_dir)
        session.queue_injection(InjectionRequest(text="abc", mode="dynamic_runtime", wrap_system_tags=False))

        for expected_count in (1, 2, 3):
            executed = session.consume_one_tick()
            self.assertTrue(executed)
            self.assertEqual(len(lm_gen.calls), expected_count)

        # Job finished on the 3rd tick -> pending_job cleared, and a completion record logged.
        self.assertIsNone(session.pending_job)
        rows = session.logger.read_all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["injected_token_count"], 3)

    def test_never_calls_reset_streaming_during_incremental_injection(self):
        session, lm_gen = _make_session(self.config, self.tmp_dir)
        session.queue_injection(InjectionRequest(text="hello", mode="dynamic_runtime"))
        while session.pending_job is not None:
            session.consume_one_tick()
        self.assertFalse(lm_gen.reset_streaming_called)


if __name__ == "__main__":
    unittest.main()
