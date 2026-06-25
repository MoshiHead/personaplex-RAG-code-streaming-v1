"""
Benchmark measurement helpers (Phase 8 of the project plan).

Scoped, for this increment, to exactly the three latency measurements needed to validate Mode C:
retrieval latency, injection latency, and total response latency. Quality metrics (context
utilization, factual grounding, hallucination rate) and memory metrics (VRAM/RAM/cache size) are
deferred to the increment where Modes B/D/E/F exist to compare against -- a single mode has nothing
to be benchmarked relative to yet beyond "did it work and how slow was it," which is what this
module measures.
"""

from __future__ import annotations

from dataclasses import dataclass
import statistics


@dataclass
class TurnBenchmark:
    """One conversational turn's timing breakdown."""

    mode: str
    retrieval_latency_s: float = 0.0
    injection_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    injected_token_count: int = 0
    retrieved_doc_count: int = 0

    @property
    def total_latency_s(self) -> float:
        return self.retrieval_latency_s + self.injection_latency_s + self.generation_latency_s


def from_log_record(record: dict) -> TurnBenchmark:
    """Build a TurnBenchmark from one `rag.logging_utils.RequestLogRecord.to_dict()` row, e.g.
    when loading `requests.jsonl` back in a notebook for the benchmark report."""
    return TurnBenchmark(
        mode=record.get("mode", "unknown"),
        retrieval_latency_s=record.get("retrieval_latency_s") or 0.0,
        injection_latency_s=record.get("injection_latency_s") or 0.0,
        generation_latency_s=record.get("generation_latency_s") or 0.0,
        injected_token_count=record.get("injected_token_count", 0),
        retrieved_doc_count=len(record.get("retrieved_contexts") or []),
    )


def _stats(values: list) -> dict:
    if not values:
        return {"mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "min_s": 0.0, "max_s": 0.0}
    values = sorted(values)
    n = len(values)
    return {
        "mean_s": round(statistics.fmean(values), 4),
        "p50_s": round(values[n // 2], 4),
        "p95_s": round(values[min(n - 1, int(n * 0.95))], 4),
        "min_s": round(values[0], 4),
        "max_s": round(values[-1], 4),
    }


def summarize(benchmarks: list) -> dict:
    """Aggregate a list of `TurnBenchmark` into mean/p50/p95 per latency field -- the minimum a
    Phase 8 report needs to say anything beyond "it ran once"."""
    if not benchmarks:
        return {"n": 0}

    return {
        "n": len(benchmarks),
        "retrieval_latency": _stats([b.retrieval_latency_s for b in benchmarks]),
        "injection_latency": _stats([b.injection_latency_s for b in benchmarks]),
        "generation_latency": _stats([b.generation_latency_s for b in benchmarks]),
        "total_latency": _stats([b.total_latency_s for b in benchmarks]),
        "mean_injected_tokens": round(statistics.fmean([b.injected_token_count for b in benchmarks]), 2),
        "mean_retrieved_docs": round(statistics.fmean([b.retrieved_doc_count for b in benchmarks]), 2),
    }
