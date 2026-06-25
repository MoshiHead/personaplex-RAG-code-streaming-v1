"""Unit tests for rag.logging_utils. Pure stdlib -- no torch/faiss/etc. required."""

import shutil
import tempfile
import unittest

from rag.logging_utils import RequestLogger, RequestLogRecord, inspect_kv_cache


class FakeLMGenWithoutStreamingState:
    """Simulates an LMGen that hasn't entered streaming mode yet."""

    _streaming_state = None


class FakeLMGenMissingInternals:
    """Simulates an object that satisfies StepCapable but doesn't expose the same private
    attribute chain real LMGen does (e.g. a future moshi version that refactors internals)."""

    _streaming_state = object()  # truthy, but has no .offset etc.


class TestInspectKvCache(unittest.TestCase):
    def test_returns_unavailable_when_not_streaming(self):
        result = inspect_kv_cache(FakeLMGenWithoutStreamingState())
        self.assertFalse(result["available"])
        self.assertIn("not currently streaming", result["reason"])

    def test_returns_unavailable_rather_than_raising_on_missing_attributes(self):
        # This must never raise -- a logging call crashing a live connection would be far worse
        # than a missing log field.
        result = inspect_kv_cache(FakeLMGenMissingInternals())
        self.assertFalse(result["available"])
        self.assertIn("introspection failed", result["reason"])


class TestRequestLogger(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.logger = RequestLogger(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_log_then_read_all_round_trips(self):
        record = RequestLogRecord(
            mode="persona_rag",
            user_query="What is the deposit?",
            retrieved_contexts=["A $300 deposit is required."],
            retrieved_scores=[0.91],
            injected_token_count=42,
        )
        self.logger.log(record)

        rows = self.logger.read_all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "persona_rag")
        self.assertEqual(rows[0]["user_query"], "What is the deposit?")
        self.assertEqual(rows[0]["injected_token_count"], 42)

    def test_multiple_records_append_in_order(self):
        for i in range(3):
            self.logger.log(RequestLogRecord(mode="persona_rag", user_query=f"q{i}"))
        rows = self.logger.read_all()
        self.assertEqual([r["user_query"] for r in rows], ["q0", "q1", "q2"])

    def test_read_all_on_missing_file_returns_empty_list(self):
        other_logger = RequestLogger(tempfile.mkdtemp())
        self.assertEqual(other_logger.read_all(), [])


if __name__ == "__main__":
    unittest.main()
