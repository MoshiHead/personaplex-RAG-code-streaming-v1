"""Unit tests for rag.benchmark. Pure stdlib -- no torch/faiss/etc. required."""

import unittest

from rag.benchmark import TurnBenchmark, from_log_record, summarize


class TestTurnBenchmark(unittest.TestCase):
    def test_total_latency_sums_all_stages(self):
        b = TurnBenchmark(mode="persona_rag", retrieval_latency_s=0.1, injection_latency_s=0.4, generation_latency_s=0.05)
        self.assertAlmostEqual(b.total_latency_s, 0.55)


class TestFromLogRecord(unittest.TestCase):
    def test_builds_benchmark_from_record_dict(self):
        record = {
            "mode": "persona_rag",
            "retrieval_latency_s": 0.12,
            "injection_latency_s": 0.34,
            "generation_latency_s": None,
            "injected_token_count": 17,
            "retrieved_contexts": ["a", "b"],
        }
        b = from_log_record(record)
        self.assertEqual(b.mode, "persona_rag")
        self.assertEqual(b.retrieval_latency_s, 0.12)
        self.assertEqual(b.injection_latency_s, 0.34)
        self.assertEqual(b.generation_latency_s, 0.0)  # None coerced to 0.0
        self.assertEqual(b.injected_token_count, 17)
        self.assertEqual(b.retrieved_doc_count, 2)


class TestSummarize(unittest.TestCase):
    def test_empty_list_returns_zero_count(self):
        self.assertEqual(summarize([]), {"n": 0})

    def test_aggregates_mean_and_percentiles(self):
        benchmarks = [
            TurnBenchmark(mode="persona_rag", retrieval_latency_s=0.1, injection_latency_s=0.2, injected_token_count=10),
            TurnBenchmark(mode="persona_rag", retrieval_latency_s=0.2, injection_latency_s=0.4, injected_token_count=20),
            TurnBenchmark(mode="persona_rag", retrieval_latency_s=0.3, injection_latency_s=0.6, injected_token_count=30),
        ]
        summary = summarize(benchmarks)

        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["retrieval_latency"]["mean_s"], 0.2)
        self.assertAlmostEqual(summary["mean_injected_tokens"], 20.0)
        self.assertIn("p50_s", summary["total_latency"])
        self.assertIn("p95_s", summary["total_latency"])


if __name__ == "__main__":
    unittest.main()
