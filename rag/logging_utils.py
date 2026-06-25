"""
Per-request structured logging (Phase 9 of the project plan).

Every record is one JSON line (JSONL), so logs are easy to tail, grep, and load straight into a
notebook with `pandas.read_json(path, lines=True)` for the Phase 8 benchmark report.

`inspect_kv_cache()` is read-only, best-effort introspection of moshi's internal streaming state
(see docs/STREAMING_AND_INJECTION_DESIGN.md, Section 2) purely for observability. It reaches into
attributes (`_streaming_state`, `.kv_cache`, ...) that are not part of moshi's stable public API,
so every access is wrapped in a single `try/except AttributeError` and degrades to
`{"available": False, ...}` rather than raising -- a log call must never be able to crash a live
conversation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import time
from typing import Any, Optional


def inspect_kv_cache(lm_gen: Any, frame_rate_hz: float = 12.5) -> dict:
    """Best-effort, read-only snapshot of the live attention KV-cache's fill state.

    Reads the first main-transformer layer's `RingKVCache` (every layer shares the same
    `end_offset` cadence -- one append per `lm_gen.step()` call -- so the first layer is
    representative of the whole stack; see docs/STREAMING_AND_INJECTION_DESIGN.md Section 2).
    """
    try:
        lm_state = lm_gen._streaming_state
        if lm_state is None:
            return {"available": False, "reason": "lm_gen is not currently streaming"}

        first_layer = lm_gen.lm_model.transformer.layers[0]
        mha_state = first_layer.self_attn._streaming_state
        kv_cache = mha_state.kv_cache

        capacity = int(kv_cache.capacity)
        end_offset = int(kv_cache.end_offset.item())
        frames_used = min(end_offset, capacity)

        return {
            "available": True,
            "lm_gen_offset": int(lm_state.offset),
            "attention_capacity_frames": capacity,
            "attention_end_offset": end_offset,
            "attention_frames_used": frames_used,
            "attention_fill_fraction": round(frames_used / capacity, 4) if capacity else None,
            "attention_window_seconds": round(capacity / frame_rate_hz, 1) if capacity else None,
            "attention_seconds_elapsed": round(end_offset / frame_rate_hz, 1),
        }
    except AttributeError as exc:
        return {"available": False, "reason": f"introspection failed: {exc!r}"}


@dataclass
class RequestLogRecord:
    """One row in the Phase 9 per-request log. Field names map directly onto the project brief's
    list (user query, retrieved context, injection strategy, prompt length, context length, KV
    cache status, generation time, final answer), plus a few extra fields the Phase 8 benchmark
    report needs (timestamps, per-stage latency, token counts)."""

    timestamp: float = field(default_factory=time.time)
    mode: str = "baseline"
    user_query: Optional[str] = None
    retrieved_contexts: list = field(default_factory=list)
    retrieved_scores: list = field(default_factory=list)
    injection_strategy: str = "none"
    prompt_length_chars: int = 0
    context_length_chars: int = 0
    injected_token_count: int = 0
    retrieval_latency_s: Optional[float] = None
    injection_latency_s: Optional[float] = None
    generation_latency_s: Optional[float] = None
    total_latency_s: Optional[float] = None
    kv_cache_status: dict = field(default_factory=dict)
    final_answer: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class RequestLogger:
    """Appends `RequestLogRecord`s to `<log_dir>/requests.jsonl`.

    Safe to share across an entire connection/session without a lock: per the concurrency contract
    documented for `TokenInjector` (docs/STREAMING_AND_INJECTION_DESIGN.md Section 3.1), only one
    coroutine ever drives RAG logic for a given connection, so log calls are never concurrent
    either.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "requests.jsonl")

    def log(self, record: RequestLogRecord) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def read_all(self) -> list:
        """Convenience for notebook/benchmark cells: load every record back as plain dicts."""
        if not os.path.exists(self.log_path):
            return []
        records = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
