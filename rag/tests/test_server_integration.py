"""
Unit tests for rag.server_integration.RAGSession, using the same plain-Python `FakeLMGen`/
`FakeTokenizer` pattern as rag/tests/test_injection_manager.py, plus a hand-rolled fake retriever
(no faiss/sentence-transformers needed) so these tests run anywhere, independent of optional RAG
dependencies being installed.
"""

import asyncio
import shutil
import tempfile
import unittest

import numpy as np

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

    def test_inject_does_not_write_to_the_log_until_finalized(self):
        retriever = FakeRetriever({"query": "q", "contexts": ["fact one"], "scores": [0.8]})
        session, _ = _make_session(self.config, self.tmp_dir, retriever=retriever)
        session.inject_persona_compatible_knowledge("a question")

        self.assertEqual(session.logger.read_all(), [])

    def test_finalize_and_log_persists_exactly_one_row(self):
        retriever = FakeRetriever({"query": "q", "contexts": ["fact one"], "scores": [0.8]})
        session, _ = _make_session(self.config, self.tmp_dir, retriever=retriever)
        record = session.inject_persona_compatible_knowledge("a question")
        session.finalize_and_log(record)

        rows = session.logger.read_all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["user_query"], "a question")

    def test_finalize_and_log_without_generation_args_leaves_those_fields_none(self):
        # Mirrors server.py's call site: no bounded generation phase to time.
        retriever = FakeRetriever({"query": "q", "contexts": ["fact one"], "scores": [0.8]})
        session, _ = _make_session(self.config, self.tmp_dir, retriever=retriever)
        record = session.inject_persona_compatible_knowledge("a question")
        finalized = session.finalize_and_log(record)

        self.assertIsNone(finalized["generation_latency_s"])
        self.assertIsNone(finalized["final_answer"])
        # total_latency_s should still equal retrieval + injection (generation contributes 0).
        expected_total = finalized["retrieval_latency_s"] + finalized["injection_latency_s"]
        self.assertAlmostEqual(finalized["total_latency_s"], expected_total)

    def test_finalize_and_log_with_generation_args_populates_and_sums_latency(self):
        # Mirrors offline.py's call site: a bounded generation phase was timed.
        retriever = FakeRetriever({"query": "q", "contexts": ["fact one"], "scores": [0.8]})
        session, _ = _make_session(self.config, self.tmp_dir, retriever=retriever)
        record = session.inject_persona_compatible_knowledge("a question")
        finalized = session.finalize_and_log(
            record, generation_latency_s=2.5, final_answer="Hello there."
        )

        self.assertEqual(finalized["generation_latency_s"], 2.5)
        self.assertEqual(finalized["final_answer"], "Hello there.")
        expected_total = (
            finalized["retrieval_latency_s"] + finalized["injection_latency_s"] + 2.5
        )
        self.assertAlmostEqual(finalized["total_latency_s"], expected_total)

        rows = session.logger.read_all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["final_answer"], "Hello there.")

    def test_finalize_and_log_works_for_skipped_records_too(self):
        # The "skipped" early-return paths from inject_persona_compatible_knowledge must also be
        # finalize-able (e.g. offline.py always calls finalize_and_log regardless of outcome).
        config = RAGConfig(enable_rag=False, log_dir=self.tmp_dir)
        session, _ = _make_session(config, self.tmp_dir, retriever=FakeRetriever({"query": "q", "contexts": [], "scores": []}))
        record = session.inject_persona_compatible_knowledge("a question")
        finalized = session.finalize_and_log(record, generation_latency_s=1.0, final_answer="hi")

        self.assertIn("skipped", finalized["injection_strategy"])
        self.assertEqual(finalized["generation_latency_s"], 1.0)
        self.assertEqual(session.logger.read_all()[0]["final_answer"], "hi")


class TestRAGSessionModeB(unittest.TestCase):
    """Mode B: the naive 'Relevant Knowledge: ... User Question: ...' negative control."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = RAGConfig(
            enable_rag=True, injection_mode=InjectionMode.PROMPT_RAG, log_dir=self.tmp_dir
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_disabled_rag_skips_injection(self):
        config = RAGConfig(enable_rag=False, log_dir=self.tmp_dir)
        retriever = FakeRetriever({"query": "q", "contexts": [], "scores": []})
        session, lm_gen = _make_session(config, self.tmp_dir, retriever=retriever)

        result = session.inject_standard_prompt_rag("What is the deposit?")

        self.assertIn("skipped", result["injection_strategy"])
        self.assertEqual(len(lm_gen.calls), 0)

    def test_empty_retrieval_result_skips_injection(self):
        retriever = FakeRetriever({"query": "q", "contexts": [], "scores": []})
        session, lm_gen = _make_session(self.config, self.tmp_dir, retriever=retriever)
        result = session.inject_standard_prompt_rag("What is the deposit?")
        self.assertIn("skipped (no contexts", result["injection_strategy"])
        self.assertEqual(len(lm_gen.calls), 0)

    def test_injected_text_matches_the_naive_template_exactly(self):
        retriever = FakeRetriever(
            {"query": "q", "contexts": ["A $300 deposit is required."], "scores": [0.92]}
        )
        session, lm_gen = _make_session(self.config, self.tmp_dir, retriever=retriever)

        session.inject_standard_prompt_rag("How much is the deposit?")

        forced_text = "".join(chr(c["text_token"]) for c in lm_gen.calls)
        expected = (
            "Relevant Knowledge:\nA $300 deposit is required.\n\n"
            "User Question:\nHow much is the deposit?\n\n"
            "Use the knowledge above when answering."
        )
        self.assertEqual(forced_text, expected)

    def test_naive_template_is_not_wrapped_in_system_tags(self):
        retriever = FakeRetriever({"query": "q", "contexts": ["fact"], "scores": [0.9]})
        session, lm_gen = _make_session(self.config, self.tmp_dir, retriever=retriever)

        session.inject_standard_prompt_rag("a question")

        forced_text = "".join(chr(c["text_token"]) for c in lm_gen.calls)
        self.assertNotIn("<system>", forced_text)

    def test_never_calls_reset_streaming(self):
        retriever = FakeRetriever({"query": "q", "contexts": ["fact"], "scores": [0.9]})
        session, lm_gen = _make_session(self.config, self.tmp_dir, retriever=retriever)
        session.inject_standard_prompt_rag("a question")
        self.assertFalse(lm_gen.reset_streaming_called)

    def test_logs_with_prompt_rag_mode_label(self):
        retriever = FakeRetriever({"query": "q", "contexts": ["fact"], "scores": [0.9]})
        session, _ = _make_session(self.config, self.tmp_dir, retriever=retriever)
        record = session.inject_standard_prompt_rag("a question")
        session.finalize_and_log(record)

        rows = session.logger.read_all()
        self.assertEqual(rows[0]["mode"], "prompt_rag")
        self.assertIn("negative control", rows[0]["injection_strategy"])


class TestModeBAndModeCRetrieveIdentically(unittest.TestCase):
    """Both connection-start modes must retrieve the same way -- the experiment is only valid if
    the *only* difference between B and C is the injection template, not the retrieval call."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = RAGConfig(enable_rag=True, log_dir=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_both_modes_call_retrieve_context_with_the_same_arguments(self):
        retriever = FakeRetriever(
            {"query": "q", "contexts": ["doc one", "doc two"], "scores": [0.9, 0.8]}
        )
        session_b, _ = _make_session(self.config, self.tmp_dir, retriever=retriever)
        session_b.inject_standard_prompt_rag("shared question")

        retriever_c = FakeRetriever(
            {"query": "q", "contexts": ["doc one", "doc two"], "scores": [0.9, 0.8]}
        )
        session_c, _ = _make_session(self.config, self.tmp_dir, retriever=retriever_c)
        session_c.inject_persona_compatible_knowledge("shared question")

        self.assertEqual(retriever.queries_seen, ["shared question"])
        self.assertEqual(retriever_c.queries_seen, ["shared question"])

    def test_modes_diverge_only_in_injected_text_not_in_retrieved_content(self):
        contexts = ["A $300 deposit is required."]
        retriever_b = FakeRetriever({"query": "q", "contexts": contexts, "scores": [0.9]})
        retriever_c = FakeRetriever({"query": "q", "contexts": contexts, "scores": [0.9]})

        session_b, lm_gen_b = _make_session(self.config, self.tmp_dir, retriever=retriever_b)
        session_c, lm_gen_c = _make_session(self.config, self.tmp_dir, retriever=retriever_c)

        record_b = session_b.inject_standard_prompt_rag("How much is the deposit?")
        record_c = session_c.inject_persona_compatible_knowledge("How much is the deposit?")

        # Same retrieved facts...
        self.assertEqual(record_b["retrieved_contexts"], record_c["retrieved_contexts"])
        self.assertEqual(record_b["retrieved_scores"], record_c["retrieved_scores"])
        # ...but different injected text (different template/wrapping) and therefore, generally,
        # a different forced-token count.
        text_b = "".join(chr(c["text_token"]) for c in lm_gen_b.calls)
        text_c = "".join(chr(c["text_token"]) for c in lm_gen_c.calls)
        self.assertNotEqual(text_b, text_c)
        self.assertTrue(text_c.startswith("<system>"))
        self.assertFalse(text_b.startswith("<system>"))


def _speech_frame(n=1920, amplitude=0.5, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n).astype(np.float32)) * amplitude


def _silence_frame(n=1920):
    return np.zeros(n, dtype=np.float32)


class TestRAGSessionModeD(unittest.TestCase):
    """Mode D: turn-boundary-triggered BURST injection (redesigned after a real run showed
    incremental per-tick interleaving corrupts both the transcript and the spoken audio -- see
    docs/MODE_D_REDESIGN.md). `observe_user_frame()` only detects; the caller fires the burst."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = RAGConfig(
            enable_rag=True,
            injection_mode=InjectionMode.TURN_INJECTION,
            vad_enabled=True,
            turn_injection_top_k=2,
            log_dir=self.tmp_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _speak_then_pause(self, session, speech_frames=3, silence_frames=16):
        # silence_frames must exceed TurnDetectorConfig's default silence_hangover_frames (15,
        # i.e. ~1.2s) -- see rag/turn_detector.py for why that default is calibrated this high.
        fired = []
        for _ in range(speech_frames):
            fired.append(session.observe_user_frame(_speech_frame()))
        for _ in range(silence_frames):
            fired.append(session.observe_user_frame(_silence_frame()))
        return fired

    def test_vad_disabled_means_no_turn_detector_and_observe_is_a_noop(self):
        config = RAGConfig(enable_rag=True, injection_mode=InjectionMode.TURN_INJECTION, vad_enabled=False, log_dir=self.tmp_dir)
        retriever = FakeRetriever({"query": "q", "contexts": ["fact"], "scores": [0.9]})
        session, lm_gen = _make_session(config, self.tmp_dir, retriever=retriever)

        record = session.prepare_turn_injection_knowledge("a question")
        session.finalize_and_log(record)
        self.assertIsNone(session.turn_detector)

        fired = self._speak_then_pause(session)
        self.assertFalse(any(fired))
        self.assertEqual(len(lm_gen.calls), 0)

    def test_observe_user_frame_is_a_noop_before_knowledge_is_prepared(self):
        retriever = FakeRetriever({"query": "q", "contexts": ["fact"], "scores": [0.9]})
        session, lm_gen = _make_session(self.config, self.tmp_dir, retriever=retriever)
        # Note: prepare_turn_injection_knowledge() was never called.
        fired = self._speak_then_pause(session)
        self.assertFalse(any(fired))
        self.assertEqual(len(lm_gen.calls), 0)

    def test_observe_user_frame_detects_boundary_but_does_not_inject_itself(self):
        retriever = FakeRetriever({"query": "q", "contexts": ["A $300 deposit is required."], "scores": [0.9]})
        session, lm_gen = _make_session(self.config, self.tmp_dir, retriever=retriever)
        record = session.prepare_turn_injection_knowledge("a question")
        session.finalize_and_log(record)

        fired = self._speak_then_pause(session)
        self.assertEqual(sum(fired), 1)
        # Detecting a boundary must not, by itself, force anything through the model -- only
        # fire_turn_injection_burst()/_async() does that.
        self.assertEqual(len(lm_gen.calls), 0)

    def test_prepare_uses_turn_injection_top_k_not_top_k(self):
        seen_top_k = []

        class RecordingRetriever(FakeRetriever):
            def retrieve_context(self, query, top_k=5, score_threshold=None, metadata_filter=None):
                seen_top_k.append(top_k)
                return super().retrieve_context(query, top_k, score_threshold, metadata_filter)

        config = RAGConfig(
            enable_rag=True, injection_mode=InjectionMode.TURN_INJECTION, vad_enabled=True,
            top_k=5, turn_injection_top_k=2, log_dir=self.tmp_dir,
        )
        retriever = RecordingRetriever({"query": "q", "contexts": ["fact"], "scores": [0.9]})
        session, _ = _make_session(config, self.tmp_dir, retriever=retriever)
        session.prepare_turn_injection_knowledge("a question")

        self.assertEqual(seen_top_k, [2])

    def test_fire_turn_injection_burst_forces_all_tokens_and_logs_one_row(self):
        retriever = FakeRetriever({"query": "q", "contexts": ["ab"], "scores": [0.9]})
        session, lm_gen = _make_session(self.config, self.tmp_dir, retriever=retriever)
        record = session.prepare_turn_injection_knowledge("a question")
        session.finalize_and_log(record)

        fired = self._speak_then_pause(session)
        self.assertEqual(sum(fired), 1)

        result = session.fire_turn_injection_burst()

        self.assertEqual(result["mode"], "turn_injection")
        self.assertEqual(result["injection_strategy"], "turn_injection (burst, fired on detected turn boundary)")
        self.assertGreater(len(lm_gen.calls), 0)
        self.assertEqual(result["injected_token_count"], len(lm_gen.calls))
        self.assertFalse(lm_gen.reset_streaming_called)

        rows = session.logger.read_all()
        burst_rows = [r for r in rows if r["injection_strategy"].startswith("turn_injection (burst")]
        self.assertEqual(len(burst_rows), 1)

    def test_can_fire_again_after_a_later_boundary(self):
        retriever = FakeRetriever({"query": "q", "contexts": ["x"], "scores": [0.9]})
        session, lm_gen = _make_session(self.config, self.tmp_dir, retriever=retriever)
        record = session.prepare_turn_injection_knowledge("a question")
        session.finalize_and_log(record)

        self._speak_then_pause(session)
        session.fire_turn_injection_burst()
        first_cycle_calls = len(lm_gen.calls)

        fired_again = self._speak_then_pause(session)
        self.assertEqual(sum(fired_again), 1)
        session.fire_turn_injection_burst()
        self.assertGreater(len(lm_gen.calls), first_cycle_calls)

        rows = session.logger.read_all()
        burst_rows = [r for r in rows if r["injection_strategy"].startswith("turn_injection (burst")]
        self.assertEqual(len(burst_rows), 2)


class TestRAGSessionModeDAsyncBurst(unittest.IsolatedAsyncioTestCase):
    """Async-checkpointed equivalent of the burst, for moshi.server's opus_loop."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = RAGConfig(
            enable_rag=True, injection_mode=InjectionMode.TURN_INJECTION, vad_enabled=True,
            turn_injection_top_k=2, log_dir=self.tmp_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_async_burst_forces_tokens_and_logs_same_as_sync(self):
        retriever = FakeRetriever({"query": "q", "contexts": ["A $300 deposit is required."], "scores": [0.9]})
        session, lm_gen = _make_session(self.config, self.tmp_dir, retriever=retriever)
        record = session.prepare_turn_injection_knowledge("a question")
        session.finalize_and_log(record)

        result = await session.fire_turn_injection_burst_async()

        self.assertEqual(result["injection_strategy"], "turn_injection (burst, fired on detected turn boundary)")
        self.assertGreater(len(lm_gen.calls), 0)
        self.assertFalse(lm_gen.reset_streaming_called)

    async def test_async_burst_does_not_starve_other_coroutines(self):
        retriever = FakeRetriever({"query": "q", "contexts": ["a somewhat longer fact to inject here"], "scores": [0.9]})
        session, _ = _make_session(self.config, self.tmp_dir, retriever=retriever)
        record = session.prepare_turn_injection_knowledge("a question")
        session.finalize_and_log(record)

        other_task_ticks = []

        async def other_task():
            for i in range(5):
                other_task_ticks.append(i)
                await asyncio.sleep(0)

        await asyncio.gather(session.fire_turn_injection_burst_async(), other_task())
        self.assertEqual(other_task_ticks, [0, 1, 2, 3, 4])


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
